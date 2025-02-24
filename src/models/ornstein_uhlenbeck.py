#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ornstein_uhlenbeck.py
=====================
Defines an Ornstein–Uhlenbeck model for X_t, referencing an exact SDE simulator
and a direct characteristic function. Typically, OU is used for rates, spreads,
or volatility factors.

Usage
-----
    from src.models.ornstein_uhlenbeck import OrnsteinUhlenbeck
    model = OrnsteinUhlenbeck(kappa=1.5, theta=2.0, sigma=0.3)
    cf = model.characteristic_function(t=1.0, X0=1.0)
    sde = model.SDE()
    X_paths = sde.sample_paths(T=1.0, n_sims=10000, n_steps=100, X0=1.0)
"""

from typing import Callable, Dict, Any
from src.models.base_model import BaseModel
from src.characteristic_equations.ornstein_uhlenbeck_cf import ou_cf
from src.sde.ornstein_uhlenbeck_sde import OrnsteinUhlenbeckSDE

class OrnsteinUhlenbeck(BaseModel):
    """
    Ornstein–Uhlenbeck model: dX_t = kappa (theta - X_t) dt + sigma dW_t.

    Attributes in _params:
      - kappa
      - theta
      - sigma

    The characteristic function is that of a Gaussian with known mean/variance.
    The SDE is an exact-step simulator for X_t.
    """

    def __init__(self, kappa: float, theta: float, sigma: float) -> None:
        """
        Initialize the OU model.

        Parameters
        ----------
        kappa : float
            Mean reversion speed.
        theta : float
            Long-run mean.
        sigma : float
            Vol parameter.
        """
        super().__init__(model_name="OrnsteinUhlenbeck", kappa=kappa, theta=theta, sigma=sigma)

    @property
    def kappa(self) -> float:
        return self._params["kappa"]

    @property
    def theta(self) -> float:
        return self._params["theta"]

    @property
    def sigma(self) -> float:
        return self._params["sigma"]

    def validate_params(self) -> None:
        """
        Validate OU parameters.

        Raises
        ------
        ValueError
            If domain constraints are violated.
        """
        if self.kappa <= 0.0:
            raise ValueError("kappa must be > 0.")
        if self.sigma <= 0.0:
            raise ValueError("sigma must be > 0.")

    def characteristic_function(
        self, t: float, X0: float, *args, **kwargs
    ) -> Callable[[complex], complex]:
        """
        Return the OU characteristic function for X_t given initial X0.

        Parameters
        ----------
        t : float
        X0 : float

        Returns
        -------
        Callable[[complex], complex]
        """
        return ou_cf(
            t,
            X0,
            self.kappa,
            self.theta,
            self.sigma
        )

    def SDE(self) -> OrnsteinUhlenbeckSDE:
        """
        Return the SDE simulator for OU.

        Returns
        -------
        OrnsteinUhlenbeckSDE
        """
        return OrnsteinUhlenbeckSDE(
            kappa=self.kappa,
            theta=self.theta,
            sigma=self.sigma
        )

    def price_option(self, *args, **kwargs) -> float:
        """
        Placeholder for an option pricing method if we interpret X as ln(S) or rates.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError("No direct 'closed form' for OU-based calls. Implement your approach.")

    def update_params(self, new_params: Dict[str, Any]) -> None:
        """
        Update the model parameters.

        Parameters
        ----------
        new_params : dict
        """
        self._params.update(new_params)

    def __repr__(self) -> str:
        base = super().__repr__()
        return f"{base}, kappa={self.kappa}, theta={self.theta}, sigma={self.sigma}"

    def __hashable_state__(self) -> tuple:
        return (self.kappa, self.theta, self.sigma)
