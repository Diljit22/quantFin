#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
black_scholes_merton.py
=======================
Unified implementation of the Black-Scholes-Merton model.

This model implements the standard Black-Scholes assumptions for equity option pricing.
It supports:
  - Calculation of the characteristic function (via an external function).
  - Access to the SDE simulator via the SDE() method.
  - Integration with closed-form techniques (handled by separate modules, not shown here).

The model requires volatility (sigma) as its only parameter.
Calibration is not implemented because volatility is assumed to be provided.
"""

import math
from typing import Callable
import numpy as np

from src.models.base_model import BaseModel
from src.characteristic_equations.black_scholes_merton_cf import black_scholes_merton_cf
from src.sde.black_scholes_merton_sde import BlackScholesMertonSDE


class BlackScholesMerton(BaseModel):
    """
    Black-Scholes-Merton model for option pricing.

    Attributes
    ----------
    _params["sigma"] : float
        The annualized volatility (e.g., 0.2 for 20%).
    """

    def __init__(self, sigma: float) -> None:
        """
        Initialize the Black-Scholes-Merton model with the specified volatility.

        Parameters
        ----------
        sigma : float
            The annualized volatility.
        """
        super().__init__(model_name="BlackScholesMerton", sigma=sigma)

    @property
    def sigma(self) -> float:
        """
        float : Returns the stored volatility parameter.
        """
        return self._params["sigma"]

    def validate_params(self) -> None:
        """
        Validate the model parameters.

        Raises
        ------
        ValueError
            If volatility (sigma) is negative.
        """
        sigma = self._params["sigma"]
        if sigma < 0.0:
            raise ValueError("Volatility (sigma) cannot be negative.")

    def characteristic_function(
        self, t: float, spot: float, r: float, q: float
    ) -> Callable[[complex], complex]:
        """
        Return the characteristic function of the log-price under the Black-Scholes-Merton model.

        Parameters
        ----------
        t : float
            Time to maturity in years.
        spot : float
            Current spot price.
        r : float
            Risk-free interest rate.
        q : float
            Dividend yield.

        Returns
        -------
        Callable[[complex], complex]
            A function phi(u) computing the characteristic function.
        """
        return black_scholes_merton_cf(t, spot, r, q, self.sigma)

    def SDE(self) -> BlackScholesMertonSDE:
        """
        Return an instance of the BlackScholesMertonSDE simulator with the model's volatility.

        Returns
        -------
        BlackScholesMertonSDE
            The SDE simulator for the Black-Scholes-Merton model.
        """
        return BlackScholesMertonSDE(self.sigma)

    def pde(self, S: float, r: float, q: float, K: float, T: float) -> tuple:
        """
        Compute the PDE coefficients for the Black–Scholes–Merton model.

        The Black–Scholes PDE for option pricing is:
            L V = A(S) * V'' + B(S) * V' - C(S) * V,
        where the coefficients are given by:
            A(S) = 0.5 * sigma^2 * S^2,
            B(S) = (r - q) * S,
            C(S) = r.

        Parameters
        ----------
        S : float
            The current asset price.
        r : float
            The risk-free interest rate.
        q : float
            The continuous dividend yield.
        K : float
            The strike price (not used in these coefficients).
        T : float
            Time to maturity (not used in these coefficients).

        Returns
        -------
        tuple of float
            A tuple (A, B, C) representing the PDE coefficients.
        """
        A = 0.5 * self.sigma**2 * S**2
        B = (r - q) * S
        C = r
        return (A, B, C)

    def __repr__(self) -> str:
        """
        Return a string representation for debugging.

        Returns
        -------
        str
            A representation including the model name and volatility.
        """
        base = super().__repr__()
        return f"{base}, sigma={self.sigma}"

    def __hashable_state__(self) -> tuple:
        """
        Generate a hashable state for caching or comparison.

        Returns
        -------
        tuple
            A tuple containing the volatility.
        """
        return (self.sigma,)
