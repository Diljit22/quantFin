#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
european_option.py
==================
Defines the EuropeanOption class, which inherits from BaseOption.
European options can only be exercised at maturity.

This module implements a European vanilla option that computes its payoff at maturity.
"""

from src.instruments.base_option import BaseOption
from dataclasses import dataclass


@dataclass(frozen=True)
class EuropeanOption(BaseOption):
    """
    Represents a European vanilla option that can only be exercised at maturity.

    Parameters
    ----------
    strike : float
        Strike price of the option.
    maturity : float
        Time to maturity in years.
    is_call : bool
        True if the option is a call, False if it is a put.
    """

    def payoff(self, spot_price: float) -> float:
        """
        Compute the payoff at maturity for a European vanilla option.

        The payoff is calculated as the intrinsic payoff:
        - For call options: max(spot_price - strike, 0)
        - For put options: max(strike - spot_price, 0)

        Parameters
        ----------
        spot_price : float
            The underlying asset's spot price at maturity.

        Returns
        -------
        float
            The intrinsic payoff value.
        """
        return self.intrinsic_payoff(spot_price)

    def __hashable_state__(self) -> tuple:
        """
        Generate a hashable state for the EuropeanOption.

        Returns
        -------
        tuple
            A tuple containing the strike, maturity, and is_call flag.
        """
        return (self.strike, self.maturity, self.is_call)
