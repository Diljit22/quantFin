"""
MarketEnvironment
=================

Manages and retrieves market data for multiple underlyings, interest rate curves,
volatility surfaces, and dividends. It also delegate real-time or
historical data fetching to a DataProvider instance (e.g., PolygonDataProvider,
FredDataProvider).

Features
--------
- Extensible data structures to store:
  * Spot prices
  * Interest rate curves (for discounting)
  * Volatility surfaces
  * Dividend yields
- Thread-safe updates through an internal lock, allowing safe integration
  with real-time data feeds.
- Allows external data providers to refresh or retrieve values (spot, rates, vol).

Performance & HPC
-----------------
- This class primarily acts as a container and retrieval mechanism, not a CPU-
  intensive component. However, thread locks are used for concurrency safety.
- In HPC contexts (large Monte Carlo or PDE solves), ensuring minimal overhead
  in data fetches will go a long way so caching or direct references (vs repeated lookups)
is beneficial.

Examples
--------
>>> from src.market.market_environment import MarketEnvironment
>>> from src.market.rate_curve import InterestRateCurve
>>> from src.market.volatility_surface import VolatilitySurface
>>>
>>> # Create a market environment with some default data
>>> env = MarketEnvironment(
...     spot_prices={"AAPL": 150.0},
...     interest_rate_curves={"USD": InterestRateCurve({1.0: 0.025})},
...     volatility_surfaces={
...         "AAPL": VolatilitySurface({(1.0, 150): 0.20, (1.0, 160): 0.22})
...     },
...     dividends={"AAPL": 0.0075}
... )
>>> current_spot = env.get_spot_price("AAPL")  # 150.0
>>> discount_factor = env.get_discount_factor(1.0, "USD")
>>> implied_vol = env.get_volatility("AAPL", 1.0, 150)
"""

import threading
import time
from typing import Dict, Optional, Tuple, List, Any, Union
from src.io.data_provider import DataProvider
from src.market.rate_curve import InterestRateCurve
from src.market.volatility_surface import VolatilitySurface


class MarketEnvironment:
    """
    Encapsulates market data for multiple underlyings, interest rate curves,
    volatility surfaces, and dividend yields.

    Attributes
    ----------
    _data_provider : DataProvider or None
        External data provider for real-time or historical quotes.
    _spot_prices : dict of str -> float
        Mapping of underlying symbol -> current spot price.
    _interest_rate_curves : dict of str -> InterestRateCurve
        Mapping of curve_name -> InterestRateCurve.
    _volatility_surfaces : dict of str -> VolatilitySurface
        Mapping of symbol -> VolatilitySurface.
    _dividends : dict of str -> float
        Mapping of symbol -> continuous dividend yield (decimal).
    _lock : threading.Lock
        Ensures thread-safe reads and writes to internal data.
    """

    def __init__(
        self,
        data_provider: Optional[DataProvider] = None,
        spot_prices: Optional[Dict[str, float]] = None,
        interest_rate_curves: Optional[Dict[str, InterestRateCurve]] = None,
        volatility_surfaces: Optional[Dict[str, VolatilitySurface]] = None,
        dividends: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Initialize the MarketEnvironment.

        Parameters
        ----------
        data_provider : DataProvider or None, optional
            An external data provider for real-time/historical quotes.
        spot_prices : dict of str -> float, optional
            Underlying symbol -> current spot price.
            e.g., {"AAPL": 150.0}.
        interest_rate_curves : dict of str -> InterestRateCurve, optional
            Mapping of curve_name -> InterestRateCurve. e.g. {"USD": InterestRateCurve(...)}.
        volatility_surfaces : dict of str -> VolatilitySurface, optional
            Mapping of symbol -> VolatilitySurface. e.g. {"AAPL": VolatilitySurface(...)}.
        dividends : dict of str -> float, optional
            Symbol -> continuous dividend yield (decimal form). e.g., {"AAPL": 0.008}.
        """
        self._data_provider = data_provider
        self._spot_prices = spot_prices if spot_prices else {}
        self._interest_rate_curves = (
            interest_rate_curves if interest_rate_curves else {}
        )
        self._volatility_surfaces = volatility_surfaces if volatility_surfaces else {}
        self._dividends = dividends if dividends else {}
        self._lock = threading.Lock()

    def get_spot_price(self, symbol: str) -> float:
        """
        Retrieve the current spot price for the given symbol.

        1) If a DataProvider is present, try to fetch from that provider.
        2) Otherwise, look up in the local _spot_prices dictionary.

        Parameters
        ----------
        symbol : str
            The underlying symbol.

        Returns
        -------
        float
            The current spot price.

        Raises
        ------
        KeyError
            If no data is found for the symbol.
        """
        with self._lock:
            if self._data_provider is not None:
                try:
                    price = self._data_provider.get_current_price(symbol)
                    self._spot_prices[symbol] = price  # Optionally cache locally
                    return price
                except NotImplementedError:
                    pass  # fallback to local data

            # Fallback if no data provider or symbol not found by provider
            if symbol not in self._spot_prices:
                raise KeyError(f"No spot price found for symbol '{symbol}'.")
            return self._spot_prices[symbol]

    def update_spot_price(self, symbol: str, price: float) -> None:
        """
        Manually update (or set) the spot price for a given symbol in the local store.

        Parameters
        ----------
        symbol : str
            The underlying symbol.
        price : float
            The new spot price.
        """
        with self._lock:
            self._spot_prices[symbol] = price

    def get_interest_rate_curve(self, curve_name: str = "default") -> InterestRateCurve:
        """
        Return the interest rate curve associated with curve_name.

        Parameters
        ----------
        curve_name : str, default='default'
            The key/name for the curve.

        Returns
        -------
        InterestRateCurve
            The requested interest rate curve.

        Raises
        ------
        KeyError
            If no curve is found for the given name.
        """
        with self._lock:
            if curve_name not in self._interest_rate_curves:
                raise KeyError(f"No interest rate curve found for '{curve_name}'.")
            return self._interest_rate_curves[curve_name]

    def get_interest_rate(self, t: float, curve_name: str = "default") -> float:
        """
        Retrieve an annualized interest rate for time t from the specified curve.

        Parameters
        ----------
        t : float
            Time in years.
        curve_name : str, default='default'
            The key/name for the interest rate curve.

        Returns
        -------
        float
            The interest rate for time t.
        """
        curve = self.get_interest_rate_curve(curve_name)
        return curve.get_rate(t)

    def get_discount_factor(self, t: float, curve_name: str = "default") -> float:
        """
        Retrieve a discount factor for time t from the specified interest rate curve.

        Parameters
        ----------
        t : float
            Time in years.
        curve_name : str, default='default'
            The key/name for the interest rate curve.

        Returns
        -------
        float
            The discount factor for time t.
        """
        curve = self.get_interest_rate_curve(curve_name)
        return curve.get_discount_factor(t)

    def set_interest_rate_curve(
        self, curve_name: str, curve: InterestRateCurve
    ) -> None:
        """
        Add or update an interest rate curve in this environment.

        Parameters
        ----------
        curve_name : str
            Name/key for the curve.
        curve : InterestRateCurve
            An InterestRateCurve instance.
        """
        with self._lock:
            self._interest_rate_curves[curve_name] = curve

    def get_volatility_surface(self, symbol: str) -> VolatilitySurface:
        """
        Retrieve the volatility surface object for a given underlying symbol.

        Parameters
        ----------
        symbol : str
            The symbol or key for the volatility surface (e.g., "AAPL").

        Returns
        -------
        VolatilitySurface
            The requested volatility surface.

        Raises
        ------
        KeyError
            If no surface is found for the given symbol.
        """
        with self._lock:
            if symbol not in self._volatility_surfaces:
                raise KeyError(f"No volatility surface found for symbol '{symbol}'.")
            return self._volatility_surfaces[symbol]

    def get_volatility(self, symbol: str, maturity: float, strike: float) -> float:
        """
        Retrieve implied volatility for an underlying symbol at a given maturity and strike.

        Parameters
        ----------
        symbol : str
            The underlying symbol (e.g., "AAPL", "SPX").
        maturity : float
            Time to maturity in years.
        strike : float
            The option strike price.

        Returns
        -------
        float
            The implied volatility (e.g., 0.20 for 20%).

        Raises
        ------
        KeyError
            If no volatility surface exists for the symbol.
        """
        surface = self.get_volatility_surface(symbol)
        return surface.get_vol(maturity, strike)

    def set_volatility_surface(self, symbol: str, surface: VolatilitySurface) -> None:
        """
        Add or update a volatility surface for a given symbol.

        Parameters
        ----------
        symbol : str
            The underlying symbol or key.
        surface : VolatilitySurface
            A VolatilitySurface instance.
        """
        with self._lock:
            self._volatility_surfaces[symbol] = surface

    def get_dividend_yield(self, symbol: str) -> float:
        """
        Retrieve the continuous dividend yield for a given symbol.

        Parameters
        ----------
        symbol : str
            The underlying symbol.

        Returns
        -------
        float
            The annualized continuous dividend yield (e.g., 0.02 for 2%).

        Notes
        -----
        Returns 0 if no dividend data is found, since some underlying may have none.
        """
        with self._lock:
            return self._dividends.get(symbol, 0.0)

    def set_dividend_yield(self, symbol: str, yield_: float) -> None:
        """
        Manually set or update the continuous dividend yield for a given symbol.

        Parameters
        ----------
        symbol : str
            The underlying symbol.
        yield_ : float
            The new annualized continuous dividend yield (decimal form).
        """
        with self._lock:
            self._dividends[symbol] = yield_

    def refresh_data_concurrently(
        self, symbols: List[str], interval: float = 10.0
    ) -> None:
        """
        Example method for spawning a background thread that periodically refreshes
        spot prices (and possibly other data) from the DataProvider.

        Parameters
        ----------
        symbols : list of str
            The list of symbols to update.
        interval : float, default=10.0
            Time in seconds between updates.

        Notes
        -----
        - If you want to incorporate more advanced real-time logic, expand or
          integrate event-driven callbacks here.
        - The thread runs as a daemon, so it won't block program exit.
        """

        def _background_price_updater() -> None:
            while True:
                with self._lock:
                    if self._data_provider is None:
                        break  # No data provider => stop
                    for sym in symbols:
                        try:
                            price = self._data_provider.get_current_price(sym)
                            self._spot_prices[sym] = price
                        except NotImplementedError:
                            pass
                time.sleep(interval)

        thread = threading.Thread(target=_background_price_updater, daemon=True)
        thread.start()

    @property
    def data_provider(self) -> Optional[DataProvider]:
        """
        DataProvider or None : The current data provider for fetching external data.
        """
        return self._data_provider

    @data_provider.setter
    def data_provider(self, provider: DataProvider) -> None:
        """
        Set or change the data provider at runtime.

        Parameters
        ----------
        provider : DataProvider
            The new data provider instance.
        """
        with self._lock:
            self._data_provider = provider

    def __repr__(self) -> str:
        """
        String representation for debugging.

        Returns
        -------
        str
            A summary of the environment's data contents.
        """
        with self._lock:
            return (
                f"MarketEnvironment(data_provider={self._data_provider}, "
                f"spot_prices={list(self._spot_prices.keys())}, "
                f"interest_rate_curves={list(self._interest_rate_curves.keys())}, "
                f"volatility_surfaces={list(self._volatility_surfaces.keys())}, "
                f"dividends={list(self._dividends.keys())})"
            )
