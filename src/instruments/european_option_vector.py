#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
european_option_vector.py
=========================
Defines the EuropeanOptionVector class, which inherits from BaseOption.
This class represents a vectorized European option that stores multiple strikes.
The scalar 'strike' attribute (inherited from BaseOption) is set to the first element
of the strikes array. 'maturity' and 'is_call' apply uniformly to the batch.

This module supports vectorized computation of option payoffs across a range
of strike prices.
"""

from dataclasses import dataclass, field, replace
import numpy as np
from typing import Union
from src.instruments.base_option import BaseOption


@dataclass(frozen=True)
class EuropeanOptionVector(BaseOption):
    """
    Represents a vectorized European option.

    Parameters
    ----------
    strikes : np.ndarray
        1D array of strike prices. Must not be empty.
    maturity : float
        Time-to-expiry in years (must be > 0).
    is_call : bool
        True if all options in the batch are calls, False if puts.
    """

    strikes: np.ndarray = field(default_factory=lambda: np.array([]))
    strike: float = field(init=False)
    maturity: float = 1.0
    is_call: bool = True

    def __post_init__(self) -> None:
        """
        Perform post-initialization validation and initialization.

        Sets the inherited scalar 'strike' attribute to the first element of the strikes
        array and validates that the strikes array is nonempty.

        Raises
        ------
        ValueError
            If the strikes array is empty.
        """
        if self.strikes.size == 0:
            raise ValueError("Strikes array must not be empty.")
        # Set the scalar strike attribute to the first strike value.
        object.__setattr__(self, "strike", float(self.strikes[0]))
        super().__post_init__()

    def payoff(self, spot_price: Union[float, np.ndarray]) -> np.ndarray:
        """
        Compute the vectorized payoff at maturity for the European option.

        If spot_price is a scalar, it is broadcast across all strikes. If spot_price
        is an array, the operation is performed elementwise.

        Parameters
        ----------
        spot_price : float or np.ndarray
            The underlying asset's spot price at maturity.

        Returns
        -------
        np.ndarray
            Array of payoffs computed for each strike. For call options, returns
            max(spot_price - strike, 0) for each strike; for put options, returns
            max(strike - spot_price, 0) for each strike.
        """
        K = self.strikes
        S = np.array(spot_price, ndmin=1)

        # Compute the raw payoff differences
        if self.is_call:
            raw = S[..., None] - K
        else:
            raw = K - S[..., None]
        return np.maximum(raw, 0.0)

    def with_strike(self, new_strikes: np.ndarray) -> "EuropeanOptionVector":
        """
        Create a new EuropeanOptionVector instance with an updated strikes array.

        Parameters
        ----------
        new_strikes : np.ndarray
            The new array of strike prices (each must be > 0).

        Returns
        -------
        EuropeanOptionVector
            A new instance with the updated strikes, where the scalar 'strike'
            attribute is set to the first element of new_strikes.

        Raises
        ------
        ValueError
            If any value in new_strikes is not positive.
        """
        if not np.all(new_strikes > 0):
            raise ValueError("All strike prices must be positive.")
        return replace(self, strikes=new_strikes)

    def __repr__(self) -> str:
        """
        Provide a string representation of the EuropeanOptionVector.

        Returns
        -------
        str
            A string representation displaying strikes, maturity, and the option type.
        """
        return (
            f"EuropeanOptionVector(strikes={self.strikes}, "
            f"maturity={self.maturity}, is_call={self.is_call})"
        )

    def __hashable_state__(self) -> tuple:
        """
        Generate a hashable state for the EuropeanOptionVector.

        Returns
        -------
        tuple
            A tuple containing the strikes (as a tuple), maturity, and is_call flag.
        """
        return (tuple(self.strikes), self.maturity, self.is_call)
