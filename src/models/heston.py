#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
heston.py
=========
Defines the Heston model, which extends the standard Black-Scholes approach by
allowing stochastic variance.

Parameters (intrinsic):
  - kappa, theta, sigma, rho, v0
( no direct 'r' or 'q', these are provided at simulation/pricing time )

Methods
-------
- characteristic_function
- SDE
- price_option (placeholder for closed-form solution)
- update_params
"""

from typing import Callable, Dict, Any
import math
from src.models.base_model import BaseModel
from src.characteristic_equations.heston_cf_hpc import heston_cf
from src.sde.heston_sde import HestonSDE


class Heston(BaseModel):
    """
    Heston model for stochastic volatility.

    The model’s intrinsic parameters (stored in _params) include:
      - kappa : mean-reversion speed
      - theta : long-run variance
      - sigma : vol of volatility
      - rho   : correlation
      - v0    : initial variance
    """

    def __init__(
        self, kappa: float, theta: float, sigma: float, rho: float, v0: float
    ) -> None:
        """
        Initialize the Heston model with given parameters.

        Parameters
        ----------
        kappa : float
            Mean-reversion speed.
        theta : float
            Long-run variance.
        sigma : float
            Vol of volatility.
        rho : float
            Correlation in [-1, 1].
        v0 : float
            Initial variance (v0 >= 0).
        """
        super().__init__(
            model_name="Heston", kappa=kappa, theta=theta, sigma=sigma, rho=rho, v0=v0
        )

    @property
    def kappa(self) -> float:
        """Mean-reversion speed."""
        return self._params["kappa"]

    @property
    def theta(self) -> float:
        """Long-run variance."""
        return self._params["theta"]

    @property
    def sigma(self) -> float:
        """Vol of volatility."""
        return self._params["sigma"]

    @property
    def rho(self) -> float:
        """Correlation."""
        return self._params["rho"]

    @property
    def v0(self) -> float:
        """Initial variance."""
        return self._params["v0"]

    def validate_params(self) -> None:
        """
        Validate model parameters.

        Raises
        ------
        ValueError
            If any parameter is out of range or inconsistent.
        """
        if self.sigma < 0.0:
            raise ValueError("Vol of vol (sigma) must be non-negative.")
        if self.v0 < 0.0:
            raise ValueError("Initial variance (v0) must be >= 0.")
        if not -1.0 <= self.rho <= 1.0:
            raise ValueError("Correlation (rho) must be in [-1, 1].")

    def characteristic_function(
        self, t: float, spot: float, r: float, q: float
    ) -> Callable[[complex], complex]:
        """
        Heston characteristic function of ln(S_t).

        Parameters
        ----------
        t : float
            Time to maturity.
        spot : float
            Current spot price.
        r : float
            Risk-free rate.
        q : float
            Dividend yield.

        Returns
        -------
        Callable[[complex], complex]
            The characteristic function phi(u).
        """
        return heston_cf(
            t, spot, r, q, self.v0, self.kappa, self.theta, self.sigma, self.rho
        )

    def SDE(self) -> HestonSDE:
        """
        Return an instance of the HestonSDE simulator with this model's parameters.

        Returns
        -------
        HestonSDE
            The SDE simulator for the Heston model.
        """
        return HestonSDE(
            kappa=self.kappa,
            theta=self.theta,
            sigma=self.sigma,
            rho=self.rho,
            v0=self.v0,
        )

    def price_option(self, *args, **kwargs) -> float:
        """
        Placeholder for Heston semi-closed-form solution for a European call.

        Raises
        ------
        NotImplementedError
            Because the solution is not shown here.
        """
        raise NotImplementedError("Heston closed-form pricing is not implemented.")

    def update_params(self, new_params: Dict[str, Any]) -> None:
        """
        Update the model parameters.

        Parameters
        ----------
        new_params : dict
            Keys can be 'kappa', 'theta', 'sigma', 'rho', 'v0'.
        """
        self._params.update(new_params)

    def __repr__(self) -> str:
        """
        Return a string representation of the Heston model.

        Returns
        -------
        str
            Representation including model name and key parameters.
        """
        base = super().__repr__()
        return (
            f"{base}, kappa={self.kappa}, theta={self.theta}, "
            f"sigma={self.sigma}, rho={self.rho}, v0={self.v0}"
        )

    def __hashable_state__(self) -> tuple:
        """
        Generate a hashable state for caching or comparison.

        Returns
        -------
        tuple
            (kappa, theta, sigma, rho, v0).
        """
        return (self.kappa, self.theta, self.sigma, self.rho, self.v0)
