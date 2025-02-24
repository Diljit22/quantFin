#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cev.py
======
Defines the CEV (Constant Elasticity of Variance) model.

This model uses the dynamics:
    dS = (r - q)*S dt + sigma * S^beta dW.

It provides:
  - An SDE method returning a CEVSDE instance.

Usage
-----
    from src.models.cev import CEV
    model = CEV(sigma=0.2, beta=0.5)
    sde_sim = model.SDE()
    paths = sde_sim.sample_paths(T=1.0, n_sims=10000, n_steps=100, r=0.05, q=0.0, S0=100)
"""

from typing import Callable, Dict, Any
from src.models.base_model import BaseModel
from src.sde.cev_sde import CEVSDE


class CEV(BaseModel):
    """
    Constant Elasticity of Variance (CEV) model for option pricing.

    Parameters stored in _params:
      - sigma: base volatility parameter
      - beta : elasticity exponent, often 0 < beta <= 1
    """

    def __init__(self, sigma: float, beta: float) -> None:
        """
        Initialize the CEV model.

        Parameters
        ----------
        sigma : float
            The base volatility parameter (e.g. 0.2).
        beta : float
            The elasticity exponent (often 0 < beta <= 1).
        """
        super().__init__(model_name="CEV", sigma=sigma, beta=beta)

    @property
    def sigma(self) -> float:
        """
        float : Return the base volatility parameter.
        """
        return self._params["sigma"]

    @property
    def beta(self) -> float:
        """
        float : Return the elasticity exponent.
        """
        return self._params["beta"]

    def validate_params(self) -> None:
        """
        Validate model parameters.

        Raises
        ------
        ValueError
            If sigma or beta is invalid.
        """
        if self.sigma < 0.0:
            raise ValueError("sigma must be non-negative for CEV.")
        if self.beta <= 0.0:
            raise ValueError("beta must be positive for CEV (commonly 0<beta<=1).")

    def SDE(self) -> CEVSDE:
        """
        Return an instance of the CEVSDE simulator.

        Returns
        -------
        CEVSDE
            The SDE simulator for the CEV model.
        """
        return CEVSDE(sigma=self.sigma, beta=self.beta)

    def price_option(self, *args, **kwargs) -> float:
        """
        Placeholder for an option pricing method. No closed-form solution is provided here.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError("No closed-form pricing implemented for CEV.")

    def update_params(self, new_params: Dict[str, Any]) -> None:
        """
        Update the model parameters with new values.

        Parameters
        ----------
        new_params : dict
            A dictionary containing new parameter values (e.g. {'sigma':0.25}).
        """
        self._params.update(new_params)

    def __repr__(self) -> str:
        """
        Return a string representation of the CEV model.

        Returns
        -------
        str
            Representation including the model name, sigma, and beta.
        """
        base = super().__repr__()
        return f"{base}, sigma={self.sigma}, beta={self.beta}"

    def __hashable_state__(self) -> tuple:
        """
        Generate a hashable state for caching or comparison.

        Returns
        -------
        tuple
            (sigma, beta).
        """
        return (self.sigma, self.beta)
