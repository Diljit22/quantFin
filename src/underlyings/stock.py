"""
stock.py
========

Defines the Stock class, a (thread-safe) container for equity underlyings.
It includes fields for symbol, spot price, volatility, and dividend yield,
all of which can be updated at runtime.
"""

import math
import threading
import numpy as np


class Stock:
    """
    A container class for an equity underlying that supports dynamic updates
    to spot price, volatility, and dividend yield.

    Parameters
    ----------
    symbol : str
        The stock's ticker symbol (e.g., "AAPL", "TSLA").
    spot : float
        The current spot price of the stock (must be > 0).
    volatility : float, optional
        The implied volatility (annualized as a decimal). Must be >= 0.
    dividend : float, optional
        The continuous dividend yield as a decimal (>= 0).

    Notes
    -----
    - Thread-safety is enforced using a reentrant lock (RLock). All reads
      and writes to internal fields are protected.
    """

    def __init__(
        self,
        spot: float,
        volatility: float,
        dividend: float,
        symbol: str = "N/A",
    ) -> None:
        self._lock = threading.RLock()

        # Validate inputs:
        if spot <= 0.0 or math.isinf(spot) or math.isnan(spot):
            raise ValueError("Spot price must be positive and finite.")
        if volatility is not None and (
            volatility < 0.0 or math.isinf(volatility) or math.isnan(volatility)
        ):
            raise ValueError("Volatility must be >= 0.0 and finite.")
        if dividend is not None and (
            dividend < 0.0 or math.isinf(dividend) or math.isnan(dividend)
        ):
            raise ValueError("Dividend yield must be >= 0.0 and finite.")

        self._spot = spot
        self._volatility = volatility
        self._dividend = dividend
        self._symbol = symbol

        # For discrete dividends (optional; used only for specific pricing techniques)
        self._discrete_dividend = None  # e.g., a numpy array of dividend amounts
        self._dividend_times = None  # e.g., a numpy array of payment times (years)

    @property
    def symbol(self) -> str:
        """
        str: The ticker symbol of the stock (read-only).
        """
        with self._lock:
            return self._symbol

    @property
    def spot(self) -> float:
        """
        float: The current spot price of the stock.
        """
        with self._lock:
            return self._spot

    @spot.setter
    def spot(self, new_spot: float) -> None:
        """
        Updates the spot price in a thread-safe manner.

        Parameters
        ----------
        new_spot : float
            The new spot price (must be positive).
        """
        if new_spot <= 0.0 or math.isinf(new_spot) or math.isnan(new_spot):
            raise ValueError("Spot price must be positive and finite.")
        with self._lock:
            self._spot = new_spot

    @property
    def volatility(self) -> float:
        """
        float: The annualized implied volatility as a decimal (>= 0).
        """
        with self._lock:
            return self._volatility

    @volatility.setter
    def volatility(self, new_vol: float) -> None:
        """
        Updates the implied volatility in a thread-safe manner.

        Parameters
        ----------
        new_vol : float
            The new annualized implied volatility (decimal).
        """
        if new_vol < 0.0 or math.isinf(new_vol) or math.isnan(new_vol):
            raise ValueError("Volatility must be >= 0.0 and finite.")
        with self._lock:
            self._volatility = new_vol

    @property
    def dividend(self) -> float:
        """
        float: The continuous dividend yield as a decimal (>= 0).
        """
        with self._lock:
            return self._dividend

    @dividend.setter
    def dividend(self, new_dividend: float) -> None:
        """
        Updates the continuous dividend yield in a thread-safe manner.

        Parameters
        ----------
        new_dividend : float
            The new dividend yield as a decimal.
        """
        if new_dividend < 0.0 or math.isinf(new_dividend) or math.isnan(new_dividend):
            raise ValueError("Dividend yield must be >= 0.0 and finite.")
        with self._lock:
            self._dividend = new_dividend

    def __repr__(self) -> str:
        """
        Returns a string representation for debugging.
        """
        with self._lock:
            return (
                f"{self.__class__.__name__}(symbol='{self._symbol}', "
                f"spot={self._spot:.4f}, volatility={self._volatility:.4f}, "
                f"dividend={self._dividend:.4f})"
            )

    def __hashable_state__(self) -> tuple:
        return (self.spot, self.volatility, self.div)

    # New properties for discrete dividends.
    @property
    def discrete_dividend(self) -> np.ndarray:
        """
        np.ndarray or None: The array of discrete dividend amounts.

        This property is optional and used by pricing models that need discrete dividend
        information. It can be set via the setter.
        """
        with self._lock:
            return self._discrete_dividend

    @discrete_dividend.setter
    def discrete_dividend(self, dividends: np.ndarray) -> None:
        """
        Set the discrete dividend amounts.

        Parameters
        ----------
        dividends : np.ndarray
            An array of dividend amounts.
        """
        with self._lock:
            self._discrete_dividend = np.asarray(dividends)

    @property
    def dividend_times(self) -> np.ndarray:
        """
        np.ndarray or None: The array of dividend payment times (in years).

        This property is optional and used by pricing models that require discrete dividend
        timings. It can be set via the setter.
        """
        with self._lock:
            return self._dividend_times

    @dividend_times.setter
    def dividend_times(self, times: np.ndarray) -> None:
        """
        Set the dividend payment times (in years).

        Parameters
        ----------
        times : np.ndarray
            An array of dividend payment times.
        """
        with self._lock:
            self._dividend_times = np.asarray(times)
