#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
american_option.py
==================
Defines the AmericanOption class, which inherits from BaseOption.
American options can be exercised any time up to maturity.

Notes
-----
This class is identical to EuropeanOption in its payoff computation,
but it is provided separately for clarity and end-user distinction.
"""

from src.instruments.base_option import BaseOption
from dataclasses import dataclass


@dataclass(frozen=True)
class AmericanOption(BaseOption):
    """
    Represents an American vanilla option that can be exercised any time up to
    and including maturity.

    Parameters
    ----------
    strike : float
        The strike price of the option.
    maturity : float
        The time to maturity (in years).
    is_call : bool
        True if the option is a call, False if it is a put.
    """

    def payoff(self, spot_price: float) -> float:
        """
        Compute the intrinsic payoff for an American vanilla option if exercised
        immediately. Although American options can be exercised before maturity,
        this method computes the standard intrinsic payoff at a given point in time.

        Parameters
        ----------
        spot_price : float
            The underlying asset's spot price at the time of exercise.

        Returns
        -------
        float
            For call options: max(spot_price - strike, 0)
            For put options: max(strike - spot_price, 0)
        """
        return self.intrinsic_payoff(spot_price)

    def __hashable_state__(self) -> tuple:
        """
        Generate a hashable state for the AmericanOption.

        Returns
        -------
        tuple
            A tuple containing the strike, maturity, and is_call flag.
        """
        return (self.strike, self.maturity, self.is_call)
