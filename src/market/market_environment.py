#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
market_environment.py
=====================
Defines the MarketEnvironment class, a lightweight container for market data,
currently supporting only a risk-free interest rate.

This module provides a thread-safe container for exogenous market data. Future
extensions may include yield curves, volatility surfaces, etc.
"""

import threading


class MarketEnvironment:
    """
    Container for exogenous market data.

    Parameters
    ----------
    rate : float, optional
        The annualized risk-free rate as a decimal (e.g., 0.05 for 5%).
        Defaults to 0.0.

    Notes
    -----
    - Additional market data (yield curves and volatility surfaces) to be added.
    - Thread-safety ensured via reentrant lock.
    - Negative rates are allowed.
    """

    def __init__(self, rate: float = 0.0) -> None:
        """
        Initialize a MarketEnvironment instance with a default risk-free rate.

        Parameters
        ----------
        rate : float, optional
            The initial risk-free rate (default is 0.0).
        """
        self._lock = threading.RLock()
        self._rate = rate

    @property
    def rate(self) -> float:
        """
        Get the current annualized risk-free rate.

        Returns
        -------
        float
            The risk-free rate as a decimal (e.g., 0.05 for 5% per annum).
        """
        with self._lock:
            return self._rate

    @rate.setter
    def rate(self, new_rate: float) -> None:
        """
        Set a new annualized risk-free rate in a thread-safe manner.

        Parameters
        ----------
        new_rate : float
            The new risk-free rate as a decimal (e.g., 0.05 for 5% per annum).

        Returns
        -------
        None
        """
        with self._lock:
            self._rate = new_rate

    def __repr__(self) -> str:
        """
        Return a string representation of the MarketEnvironment instance.

        Returns
        -------
        str
            A string in the format: "MarketEnvironment(rate=<value>)".
        """
        with self._lock:
            return f"{self.__class__.__name__}(rate={self._rate:.6f})"

    def __hashable_state__(self) -> tuple:
        """
        Generate a hashable state for caching or comparison purposes.

        Returns
        -------
        tuple
            A tuple containing the current risk-free rate.
        """
        return (self.rate,)
