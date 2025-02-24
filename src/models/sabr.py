#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sabr.py
=======
Defines the SABR model class, referencing a standard SDE simulator and placeholders
for characteristic function or closed-form solutions.

Commonly used for interest rates or equity with alpha(t) as stochastic volatility,
beta in [0,1], correlation rho, and vol-of-vol nu.

Usage
-----
    from src.models.sabr import SABR
    model = SABR(alpha0=0.2, beta=1.0, rho=-0.3, nu=0.5)
    sde_sim = model.SDE()
    F_paths, alpha_paths = sde_sim.sample_paths(T=1.0, n_sims=10000, n_steps=100, F0=100.0)
"""

from typing import Callable, Dict, Any
from src.models.base_model import BaseModel
from src.sde.sabr_sde import SABRSDE


class SABR(BaseModel):
    """
    SABR model for stochastic alpha (vol), F^beta drift, correlation, etc.

    Parameters
    ----------
    alpha0 : float
        Initial volatility (alpha(0)).
    beta : float
        The exponent in F^beta. Typically in [0,1].
    rho : float
        Correlation in [-1,1].
    nu : float
        Vol of vol.

    No characteristic function is provided, as SABR does not have a known closed-form CF.
    """

    def __init__(self, alpha0: float, beta: float, rho: float, nu: float) -> None:
        """
        Initialize the SABR model.

        Parameters
        ----------
        alpha0 : float
            Initial alpha(0).
        beta : float
            Elasticity exponent in [0,1].
        rho : float
            Correlation in [-1,1].
        nu : float
            Vol of volatility.
        """
        super().__init__(model_name="SABR", alpha0=alpha0, beta=beta, rho=rho, nu=nu)

    @property
    def alpha0(self) -> float:
        """Initial volatility level."""
        return self._params["alpha0"]

    @property
    def beta(self) -> float:
        """Elasticity exponent."""
        return self._params["beta"]

    @property
    def rho(self) -> float:
        """Correlation."""
        return self._params["rho"]

    @property
    def nu(self) -> float:
        """Vol of vol."""
        return self._params["nu"]

    def validate_params(self) -> None:
        """
        Validate SABR parameters.

        Raises
        ------
        ValueError
            If parameters are out of domain.
        """
        if self.alpha0 < 0.0:
            raise ValueError("alpha0 must be >= 0.")
        if not -1.0 <= self.rho <= 1.0:
            raise ValueError("rho in [-1,1].")
        if self.nu < 0.0:
            raise ValueError("nu must be >= 0.")
        # beta usually in [0,1], but not absolutely mandatory.

    def SDE(self) -> SABRSDE:
        """
        Return the SDE simulator for SABR.

        Returns
        -------
        SABRSDE
            SDE object that can simulate paths (F, alpha).
        """
        from src.sde.sabr_sde import SABRSDE

        return SABRSDE(alpha0=self.alpha0, beta=self.beta, rho=self.rho, nu=self.nu)

    def characteristic_function(self, *args, **kwargs):
        """
        No known closed-form CF for SABR.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError(
            "SABR has no simple closed-form characteristic function."
        )

    def price_option(self, *args, **kwargs):
        """
        Placeholder for a SABR-based option pricing approach (Hagan expansions, etc).

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError(
            "No direct closed-form for SABR. Try expansions or numeric methods."
        )

    def update_params(self, new_params: Dict[str, Any]) -> None:
        """
        Update model parameters with new values.

        Parameters
        ----------
        new_params : dict
            Possibly containing 'alpha0','beta','rho','nu'.
        """
        self._params.update(new_params)

    def __repr__(self) -> str:
        """
        Representation for debugging.

        Returns
        -------
        str
        """
        base = super().__repr__()
        return (
            f"{base}, alpha0={self.alpha0}, beta={self.beta}, "
            f"rho={self.rho}, nu={self.nu}"
        )

    def __hashable_state__(self) -> tuple:
        """
        Hashable state for caching.

        Returns
        -------
        tuple
            (alpha0, beta, rho, nu).
        """
        return (self.alpha0, self.beta, self.rho, self.nu)
