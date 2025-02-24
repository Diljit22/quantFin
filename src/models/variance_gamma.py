#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
variance_gamma.py
=================
Defines the Variance Gamma model class, referencing:
  - CF from variance_gamma_cf.py
  - SDE from variance_gamma_sde.py

We also provide a placeholder for direct 'price_option' or we can rely on
a separate "variance_gamma_closed_form.py" that uses a numeric integral of the CF.

Usage
-----
    from src.models.variance_gamma import VarianceGamma
    model = VarianceGamma(sigma=0.2, theta=-0.1, nu=0.3)
    cf = model.characteristic_function(t=1.0, spot=100, r=0.05, q=0.02)
    sde = model.SDE()
    paths = sde.sample_paths(T=1.0, n_sims=10000, n_steps=100, r=0.05, q=0.02, S0=100)
"""

from typing import Callable, Dict, Any
import math
from src.models.base_model import BaseModel
from src.characteristic_equations.variance_gamma_cf import variance_gamma_cf
from src.sde.variance_gamma_sde import VarianceGammaSDE


class VarianceGamma(BaseModel):
    """
    Variance Gamma model for log-price increments.

    Attributes
    ----------
    _params : dict
      - sigma : volatility scale
      - theta : drift param for the Brownian part
      - nu    : gamma scale

    The risk-neutral drift correction is done in the CF so that E[S_t] = S_0 e^{(r - q) t}.
    """

    def __init__(self, sigma: float, theta: float, nu: float) -> None:
        """
        Initialize the Variance Gamma model.

        Parameters
        ----------
        sigma : float
            Vol scale in the underlying Brownian.
        theta : float
            Drift in the underlying Brownian.
        nu : float
            Gamma scale param.
        """
        super().__init__(model_name="VarianceGamma", sigma=sigma, theta=theta, nu=nu)

    @property
    def sigma(self) -> float:
        """Vol scale."""
        return self._params["sigma"]

    @property
    def theta(self) -> float:
        """Drift in the Brownian subordinator."""
        return self._params["theta"]

    @property
    def nu(self) -> float:
        """Gamma scale param."""
        return self._params["nu"]

    def validate_params(self) -> None:
        """
        Validate parameters.

        Raises
        ------
        ValueError
            If out of domain.
        """
        if self.sigma < 0.0:
            raise ValueError("sigma must be >= 0.")
        if self.nu <= 0.0:
            raise ValueError("nu must be > 0.")

    def characteristic_function(
        self, t: float, spot: float, r: float, q: float
    ) -> Callable[[complex], complex]:
        """
        Return the characteristic function of ln(S_t).

        The CF includes the drift correction so that E[S_t] = S_0 e^{(r-q) t}.

        Parameters
        ----------
        t : float
            Time in years.
        spot : float
            Current spot price.
        r : float
            Risk-free rate.
        q : float
            Dividend yield.

        Returns
        -------
        Callable[[complex], complex]
            phi(u).
        """
        return variance_gamma_cf(t, spot, r, q, self.sigma, self.theta, self.nu)

    def SDE(self) -> VarianceGammaSDE:
        """
        Return the SDE-like simulator for the Variance Gamma process.

        Returns
        -------
        VarianceGammaSDE
        """
        return VarianceGammaSDE(sigma=self.sigma, theta=self.theta, nu=self.nu)

    def price_option(self, *args, **kwargs) -> float:
        """
        Placeholder for a numeric or semi-closed approach to price calls.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError(
            "Use a separate numeric integral approach or expansions."
        )

    def update_params(self, new_params: Dict[str, Any]) -> None:
        """
        Update the model parameters.

        Parameters
        ----------
        new_params : dict
            Possibly containing 'sigma','theta','nu'.
        """
        self._params.update(new_params)

    def __repr__(self) -> str:
        """
        Return a representation of the VarianceGamma model.

        Returns
        -------
        str
        """
        base = super().__repr__()
        return f"{base}, sigma={self.sigma}, theta={self.theta}, nu={self.nu}"

    def __hashable_state__(self) -> tuple:
        """
        Hashable state for caching.

        Returns
        -------
        tuple
        """
        return (self.sigma, self.theta, self.nu)
