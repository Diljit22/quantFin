#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
bermudan_option.py
==================
Defines the BermudanOption class, which inherits from BaseOption.
Bermudan options can be exercised on a specified set of dates up to maturity.

This module implements a Bermudan vanilla option with discrete exercise dates.
It ensures that all exercise dates are positive and do not exceed the option's maturity.
"""

import numpy as np
from typing import List
from src.instruments.base_option import BaseOption


class BermudanOption(BaseOption):
    """
    Represents a Bermudan vanilla option that can be exercised only on specified
    dates up to maturity.

    Parameters
    ----------
    strike : float
        The strike price of the option.
    maturity : float
        The time to maturity (in years).
    is_call : bool
        True if the option is a call, False if it is a put.
    exercise_dates : list of float
        The sorted list of exercise dates (in years).
        All dates must be greater than 0 and not exceed the maturity time.
    """

    def __init__(
        self, strike: float, maturity: float, is_call: bool, exercise_dates: List[float]
    ) -> None:
        super().__init__(strike, maturity, is_call)
        if not exercise_dates:
            raise ValueError("BermudanOption requires at least one exercise date.")
        # Convert the list to a sorted numpy array for consistency and performance.
        self._exercise_dates = np.sort(np.array(exercise_dates, dtype=float))
        if self._exercise_dates[0] <= 0:
            raise ValueError("Exercise dates must be positive (greater than 0).")
        if self._exercise_dates[-1] > maturity:
            raise ValueError("Exercise dates must not exceed the maturity time.")

    @property
    def exercise_dates(self) -> np.ndarray:
        """
        Get the possible exercise dates for this Bermudan option.

        Returns
        -------
        np.ndarray
            An array of exercise dates (in years).
        """
        return self._exercise_dates

    def can_exercise(self, t: float, tol: float = 1e-5) -> bool:
        """
        Determine whether the option can be exercised at time t based exercise dates.

        Parameters
        ----------
        t : float
            The time (in years) at which to check for an exercise opportunity.
        tol : float, optional
            The tolerance for floating point comparison (default is 1e-5).

        Returns
        -------
        bool
            True if t is within tol of any exercise date, False otherwise.
        """
        return np.any(np.abs(self._exercise_dates - t) < tol)

    def payoff(self, spot_price: float) -> float:
        """
        Compute the payoff at exercise for the Bermudan option.

        The payoff is defined as the intrinsic payoff:
            - For call options: max(spot_price - strike, 0)
            - For put options: max(strike - spot_price, 0)

        Parameters
        ----------
        spot_price : float
            The underlying asset's spot price at the exercise date.

        Returns
        -------
        float
            The intrinsic payoff value.
        """
        return self.intrinsic_payoff(spot_price)

    def __hashable_state__(self) -> tuple:
        """
        Generate a hashable state for the BermudanOption.

        Returns
        -------
        tuple
            A tuple containing: strike, maturity, is_call flag, tuple(exercise_dates).
        """
        return (self.strike, self.maturity, self.is_call, tuple(self._exercise_dates))
