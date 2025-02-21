"""
market_enviroment.py
====================

Defines the MarketEnvironment class, a lightweight container for
market data (currently just a risk-free interest rate).
"""

import threading


class MarketEnvironment:
    """
    A container for exogenous market data.

    Parameters
    ----------
    rate : float, optional
        The annualized risk-free rate (as a decimal).
        Defaults to 0.0.

    Notes
    -----
    - TO ADD: yield curve, market volatility surfaces.
    - Thread-safety is considered via a lock, so updates don't conflict.
    - Negative rates are allowed.
    """

    def __init__(self, rate: float = 0.0) -> None:
        self._lock = threading.RLock()
        self._rate = rate

    @property
    def rate(self) -> float:
        """
        The annualized risk-free rate (decimal). For example, 0.05 for 5%.
        """
        with self._lock:
            return self._rate

    @rate.setter
    def rate(self, new_rate: float) -> None:
        """
        Updates the risk-free rate in a thread-safe manner.

        Parameters
        ----------
        new_rate : float
            The new annualized risk-free rate (decimal). For example, 0.05
            represents 5% per annum.
        """
        with self._lock:
            self._rate = new_rate

    def __repr__(self) -> str:
        """
        String representation of the market environment for debugging.
        """
        with self._lock:
            return f"{self.__class__.__name__}(rate={self._rate:.6f})"

    def __hashable_state__(self) -> tuple:
        return (self.rate,)
