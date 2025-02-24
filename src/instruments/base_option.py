#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
base_option.py
==============
Defines the abstract base class for options. Other option types (e.g., European,
American, Bermudan) inherit from this class.

This module provides a template for vanilla options. It ensures that every option
has a strike, maturity, and a flag indicating whether it is a call or put. Additionally,
it supplies utility methods to compute intrinsic payoffs and to create modified copies
of options with updated parameters.

Usage
-----
Subclass BaseOption to implement the `payoff` method for specific option types.

Examples
--------
>>> from instruments.base_option import BaseOption
>>>
>>> class EuropeanOption(BaseOption):
...     def payoff(self, spot_price: float) -> float:
...         return self.intrinsic_payoff(spot_price)
...
>>> option = EuropeanOption(strike=100, maturity=1, is_call=True)
>>> print(option.payoff(spot_price=105))
5.0
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class BaseOption(ABC):
    """
    Abstract base class for a vanilla option.

    Parameters
    ----------
    strike : float
        Strike price of the option (must be > 0).
    maturity : float
        Time to maturity in years (must be > 0).
    is_call : bool
        True for a call, False for a put.

    Attributes
    ----------
    strike : float
        Strike price of the option (must be > 0). :noindex:
    maturity : float
        Time to maturity of the option (must be > 0). :noindex:
    is_call : bool
        Flag indicating whether the option is a call (True) or a put (False). :noindex:

    """

    strike: float
    maturity: float
    is_call: bool

    def __post_init__(self) -> None:
        """
        Perform post-initialization validation to ensure positive strike and maturity.

        Raises
        ------
        ValueError
            If `strike` or `maturity` is not positive.
        """
        if self.strike <= 0:
            raise ValueError("Strike price must be positive.")
        if self.maturity <= 0:
            raise ValueError("Time to maturity must be positive.")

    @property
    def option_type(self) -> str:
        """
        Get the option type as a readable string.

        Returns
        -------
        str
            "Call" if `is_call` is True, otherwise "Put".
        """
        return "Call" if self.is_call else "Put"

    def intrinsic_payoff(self, spot_price: float) -> float:
        """
        Compute the intrinsic payoff for a vanilla call or put.

        The intrinsic payoff is defined as max(spot_price - strike, 0) for calls,
        and max(strike - spot_price, 0) for puts.

        Parameters
        ----------
        spot_price : float
            The underlying asset's spot price at exercise/maturity.

        Returns
        -------
        float
            The intrinsic payoff value.
        """
        intrin_val = spot_price - self.strike
        return max(intrin_val, 0.0) if self.is_call else max(-intrin_val, 0.0)

    @abstractmethod
    def payoff(self, spot_price: float) -> float:
        """
        Compute the option payoff at exercise.

        Subclasses must override this method to define the specific payoff behavior.

        Parameters
        ----------
        spot_price : float
            The underlying asset's spot price at the time of exercise/maturity.

        Returns
        -------
        float
            The computed payoff value.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement the `payoff` method.")

    def with_strike(self, new_strike: float) -> "BaseOption":
        """
        Create a new option instance with an updated strike price.

        Parameters
        ----------
        new_strike : float
            The new strike price (must be > 0).

        Returns
        -------
        BaseOption
            A new option instance with the updated strike, retaining the same
            maturity and option type.

        Raises
        ------
        ValueError
            If `new_strike` is not positive.
        """
        if new_strike <= 0:
            raise ValueError("Strike price must be positive.")
        return replace(self, strike=new_strike)

    def with_maturity(self, new_maturity: float) -> "BaseOption":
        """
        Create a new option instance with an updated time to maturity.

        Parameters
        ----------
        new_maturity : float
            The new time to maturity in years (must be > 0).

        Returns
        -------
        BaseOption
            A new option instance with the updated maturity, retaining the same
            strike and option type.

        Raises
        ------
        ValueError
            If `new_maturity` is not positive.
        """
        if new_maturity <= 0:
            raise ValueError("Time to maturity must be positive.")
        return replace(self, maturity=new_maturity)

    @property
    def companion_option(self) -> "BaseOption":
        """
        Generate a companion option with the same strike and maturity but with the
        option type inverted (i.e., a call becomes a put and vice versa).

        Returns
        -------
        BaseOption
            A new option instance with the `is_call` flag inverted.
        """
        return replace(self, is_call=not self.is_call)
