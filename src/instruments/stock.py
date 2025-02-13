"""
Stock
=====

Implements the Stock class, representing a equity instrument that
inherits from BaseInstrument. The pricing of a stock alone is the current
spot price. This class stores attributes like:
- The locally tracked spot price (for offline usage).
- Dividend yield.
- Volatility.
- A price history log (for analytics).
- A reference to a MarketEnvironment for real-time spot price retrieval.

Concurrency & HPC
-----------------
- A thread lock is used to safely update the local spot price and price history,
  ensuring no race conditions occur if multiple threads write simultaneously.
- WARNING: For HPC or real-time contexts, ensure that any shared MarketEnvironment 
  usage is properly synchronized or thread-safe.

Examples
--------
>>> from src.instruments.stock import Stock
>>> from src.market.market_environment import MarketEnvironment
>>> # Create a stock with an initial spot price of 100
>>> my_stock = Stock("AAPL", maturity=1e6, initial_spot=100.0)  # Large maturity for "no expiration"
>>> # Value is just the spot unless a market_env is provided
>>> current_price = my_stock.value()
>>> print(current_price)
100.0
"""

import threading
from datetime import datetime
from typing import Optional, List, Tuple, Any
from src.instruments.base_instrument import BaseInstrument
from src.market.market_environment import MarketEnvironment


class Stock(BaseInstrument):
    """
    A Stock instrument, inheriting from BaseInstrument.

    The 'value()' method returns the current spot price as fetched from the
    MarketEnvironment if available; otherwise, it uses the locally tracked spot.

    Parameters
    ----------
    underlying_symbol : str
        Ticker or identifier for the stock (e.g., 'AAPL').
    maturity : float
        Time to maturity (in years). For a stock, this could be arbitrary if
        you do not treat it as expiring, but is required by BaseInstrument.
    market_env : MarketEnvironment or None, optional
        A reference to a MarketEnvironment for real-time spot price retrieval.
    initial_spot : float, default=100.0
        The initial or local spot price if no MarketEnvironment is available.
    dividend_yield : float, default=0.0
        The continuous annualized dividend yield (decimal form), e.g. 0.02 for 2%/year.
    volatility : float, default=0.2
        The annualized volatility (decimal), e.g., 0.20 for 20%.

    Attributes
    ----------
    _local_spot : float
        The locally maintained spot price.
    _dividend_yield : float
        The stock's annual dividend yield (continuously compounded).
    _volatility : float
        The annualized volatility of the stock.
    _lock : threading.Lock
        A lock to guard concurrent writes to `_local_spot` and `_price_history`.
    _price_history : list of (datetime, float)
        Records updates to the local spot price (timestamp, new_price).

    Raises
    ------
    ValueError
        If any invalid parameters are provided.
    """

    def __init__(
        self,
        underlying_symbol: str,
        maturity: float,
        market_env: Optional[MarketEnvironment] = None,
        initial_spot: float = 100.0,
        dividend_yield: float = 0.0,
        volatility: float = 0.2,
    ) -> None:
        super().__init__(
            underlying_symbol=underlying_symbol,
            maturity=maturity,
            market_env=market_env,
        )

        if initial_spot < 0:
            raise ValueError("initial_spot must be non-negative.")
        self._local_spot = float(initial_spot)

        if dividend_yield < 0:
            raise ValueError("dividend_yield cannot be negative.")
        self._dividend_yield = float(dividend_yield)

        if volatility < 0:
            raise ValueError("volatility cannot be negative.")
        self._volatility = float(volatility)

        # Thread lock for safe concurrent updates
        self._lock = threading.Lock()

        # simple history of (timestamp, price) for analysis or debugging
        self._price_history: List[Tuple[datetime, float]] = [
            (datetime.now(), self._local_spot)
        ]

    @property
    def dividend_yield(self) -> float:
        """
        float : The continuous annualized dividend yield of the stock (0.0 if none).
        """
        return self._dividend_yield

    @property
    def volatility(self) -> float:
        """
        float : The annualized volatility of the stock's returns.
        """
        return self._volatility

    def update_price(
        self, new_price: float, timestamp: Optional[datetime] = None
    ) -> None:
        """
        Update the locally tracked spot price, and record it in the price history.

        Parameters
        ----------
        new_price : float
            The new spot price for the stock.
        timestamp : datetime or None, optional
            The timestamp for this price update. Defaults to the current time if None.

        Raises
        ------
        ValueError
            If the new_price is negative.
        """
        if new_price < 0:
            raise ValueError("new_price cannot be negative.")

        with self._lock:
            self._local_spot = new_price
            self._price_history.append((timestamp or datetime.now(), new_price))

    def get_price_history(self) -> List[Tuple[datetime, float]]:
        """
        Retrieve a copy of the price history as a list of (timestamp, price).

        Returns
        -------
        List[Tuple[datetime, float]]
            A list of (timestamp, spot_price) entries tracking updates over time.
        """
        with self._lock:
            return list(self._price_history)

    def value(self, market_env: Any = None, **kwargs) -> float:
        """
        Return the current spot price of the stock, using the provided
        MarketEnvironment if available.

        Parameters
        ----------
        market_env : Any, optional
            A specific environment to pull the spot price from. If None,
            defaults to self.market_env. If that is also None or lacks data
            for this symbol, use the locally tracked spot.
        **kwargs :
            Additional parameters, unused by default.

        Returns
        -------
        float
            The current spot price.

        Raises
        ------
        KeyError
            If the environment is specified but does not have a spot price
            for this symbol.
        """
        if market_env is None:
            market_env = self.market_env

        if market_env is not None:
            try:
                # Attempt to fetch from the environment
                return market_env.get_spot_price(self.underlying_symbol)
            except KeyError:
                # Fallback to local spot if no data in env
                pass

        # Fallback if no environment or symbol data
        with self._lock:
            return self._local_spot

    def payoff(self, spot_price: float, **kwargs) -> float:
        """
        Stocks generally do not have a discrete payoff at maturity like an option.
        This method raises NotImplementedError by default.

        Parameters
        ----------
        spot_price : float
            Current or final price of the underlying.

        Raises
        ------
        NotImplementedError
            Because a stock is not typically exercised at maturity.
        """
        raise NotImplementedError(
            "Stock does not define a derivative payoff. Use 'value()' for the spot price."
        )

    def __repr__(self) -> str:
        """
        String representation for debugging.

        Returns
        -------
        str
            The stock's symbol, maturity, and current local spot data.
        """
        with self._lock:
            info = (
                f"symbol={self.underlying_symbol}, "
                f"maturity={self.maturity}, "
                f"spot={self._local_spot}, "
                f"dividend_yield={self._dividend_yield}, "
                f"volatility={self._volatility}"
            )
        return f"Stock({info})"
