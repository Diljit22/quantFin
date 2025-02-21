"""
bermudan_option.py
==================

Defines the BermudanOption class, which inherits from BaseOption.
Bermudan options can be exercised on a specified set of dates up to maturity.
"""

import numpy as np
from typing import List
from src.instruments.base_option import BaseOption


class BermudanOption(BaseOption):
    """
    Represents a Bermudan vanilla option. It can be exercised only on certain
    discrete dates (exercise dates) up to maturity.

    Parameters
    ----------
    strike : float
        The strike price of the option.
    maturity : float
        The time to maturity (in years).
    is_call : bool
        True if the option is a call, False if it is a put.
    exercise_dates : list of float
        The sorted list of times (in years from now) on which the holder can
        exercise this option. All must be between 0 and maturity.

    """

    def __init__(
        self, strike: float, maturity: float, is_call: bool, exercise_dates: List[float]
    ) -> None:
        super().__init__(strike, maturity, is_call)

        if not exercise_dates:
            raise ValueError("BermudanOption requires at least one exercise date.")

        self._exercise_dates = np.sort(exercise_dates)
        if self._exercise_dates[0] <= 0:
            raise ValueError("Exercise dates must be non-negative.")
        if self._exercise_dates[-1] > maturity:
            raise ValueError("Exercise dates must not exceed the maturity time.")

    @property
    def exercise_dates(self):
        """
        The np.array of possible exercise times (in years) for this Bermudan option.
        """
        return self._exercise_dates

    def can_exercise(self, t: float, tol: float = 1e-5) -> bool:
        """
        Returns True if time t (in years) is close to one of the exercise times.
        tol: tolerance level for floating point comparison.
        """
        return np.any(np.abs(self._exercise_times - t) < tol)

    def payoff(self, spot_price: float) -> float:
        """
        Computes the payoff if exercised at a known spot price. For a single
        exercise event, the payoff is the standard vanilla formula.

        Parameters
        ----------
        spot_price : float
            The spot price at the exercise date in question.

        Returns
        -------
        float
            max(S - K, 0) for calls, or max(K - S, 0) for puts.
        """
        return self.intrinsic_payoff(spot_price)
