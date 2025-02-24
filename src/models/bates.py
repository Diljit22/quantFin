#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
bates.py
========
Defines the Bates model, which merges Heston's stochastic volatility and Merton's jump-diffusion.

Intrinsic parameters in _params:
  - kappa, theta, sigma, rho, v0
  - jump_intensity, muJ, sigmaJ

References
----------
- Bates, D. (1996). "Jumps and Stochastic Volatility: Exchange Rate Processes
  Implicit in Deutsche Mark Options." The Review of Financial Studies.
- Heston, S. (1993). 
- Merton, R. (1976) "Option Pricing when Underlying Stock Returns are Discontinuous."

Usage
-----
    from src.models.bates import Bates
    model = Bates(
        kappa=1.5, theta=0.04, sigma=0.3, rho=-0.5, v0=0.04,
        jump_intensity=0.1, muJ=-0.1, sigmaJ=0.2
    )
    cf_func = model.characteristic_function(t=1.0, spot=100.0, r=0.05, q=0.02)
    sde_sim = model.SDE()
    S, v = sde_sim.sample_paths(...)
"""

from typing import Callable, Dict, Any
from src.models.base_model import BaseModel
from src.characteristic_equations.bates_cf_hpc import bates_cf_hpc
from src.sde.bates_sde import BatesSDE


class Bates(BaseModel):
    """
    Bates model = Heston SV + Merton jumps.

    Attributes
    ----------
    _params (dict) with:
      - kappa, theta, sigma, rho, v0 : Heston parameters
      - jump_intensity, muJ, sigmaJ : jump parameters
    """

    def __init__(
        self,
        kappa: float,
        theta: float,
        sigma: float,
        rho: float,
        v0: float,
        jump_intensity: float,
        muJ: float,
        sigmaJ: float,
    ) -> None:
        """
        Initialize the Bates model with Heston + jump parameters.

        Parameters
        ----------
        kappa : float
            Mean-reversion speed.
        theta : float
            Long-run variance level.
        sigma : float
            Vol of volatility.
        rho : float
            Correlation in [-1, 1].
        v0 : float
            Initial variance.
        jump_intensity : float
            Poisson jump intensity (lambda).
        muJ : float
            Mean jump size in log space.
        sigmaJ : float
            Vol of jump sizes in log space.
        """
        super().__init__(
            model_name="Bates",
            kappa=kappa,
            theta=theta,
            sigma=sigma,
            rho=rho,
            v0=v0,
            jump_intensity=jump_intensity,
            muJ=muJ,
            sigmaJ=sigmaJ,
        )

    @property
    def kappa(self) -> float:
        """Mean-reversion speed."""
        return self._params["kappa"]

    @property
    def theta(self) -> float:
        """Long-run variance level."""
        return self._params["theta"]

    @property
    def sigma(self) -> float:
        """Vol of volatility."""
        return self._params["sigma"]

    @property
    def rho(self) -> float:
        """Correlation in [-1,1]."""
        return self._params["rho"]

    @property
    def v0(self) -> float:
        """Initial variance."""
        return self._params["v0"]

    @property
    def jump_intensity(self) -> float:
        """Poisson jump intensity (lambda)."""
        return self._params["jump_intensity"]

    @property
    def muJ(self) -> float:
        """Mean jump size (log scale)."""
        return self._params["muJ"]

    @property
    def sigmaJ(self) -> float:
        """Vol of jump sizes (log scale)."""
        return self._params["sigmaJ"]

    def validate_params(self) -> None:
        """
        Validate the model parameters.

        Raises
        ------
        ValueError
            If any parameter is out of range or inconsistent.
        """
        if self.sigma < 0.0:
            raise ValueError("sigma (vol of vol) must be >= 0.")
        if self.v0 < 0.0:
            raise ValueError("initial variance v0 must be >= 0.")
        if not -1.0 <= self.rho <= 1.0:
            raise ValueError("rho must be in [-1,1].")
        if self.jump_intensity < 0.0:
            raise ValueError("jump_intensity must be >= 0.")

    def characteristic_function(
        self, t: float, spot: float, r: float, q: float, trap: int = 1
    ) -> Callable[[complex], complex]:
        """
        Return the HPC-style characteristic function for the Bates model.

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
        trap : int, optional
            1 => "Little Trap" HPC approach, 0 => original approach.

        Returns
        -------
        Callable[[complex], complex]
            phi(u), a function computing the CF at complex u.
        """
        from src.characteristic_equations.bates_cf_hpc import bates_cf_hpc

        return bates_cf_hpc(
            t,
            spot,
            r,
            q,
            self.v0,
            self.kappa,
            self.theta,
            self.sigma,
            self.rho,
            self.jump_intensity,
            self.muJ,
            self.sigmaJ,
            trap=trap,
        )

    def SDE(self) -> BatesSDE:
        """
        Return an instance of the BatesSDE simulator with model parameters.

        Returns
        -------
        BatesSDE
            SDE simulator for the Bates model.
        """
        return BatesSDE(
            kappa=self.kappa,
            theta=self.theta,
            sigma=self.sigma,
            rho=self.rho,
            v0=self.v0,
            jump_intensity=self.jump_intensity,
            muJ=self.muJ,
            sigmaJ=self.sigmaJ,
        )

    def price_option(self, *args, **kwargs) -> float:
        """
        Placeholder for a semi-closed-form or numerical approach to option pricing.

        Raises
        ------
        NotImplementedError
            Not implemented here.
        """
        raise NotImplementedError("No closed-form pricing for Bates implemented.")

    def update_params(self, new_params: Dict[str, Any]) -> None:
        """
        Update the model's parameters with new values.

        Parameters
        ----------
        new_params : dict
            Possibly containing 'kappa','theta','sigma','rho','v0','jump_intensity','muJ','sigmaJ'.
        """
        self._params.update(new_params)

    def __repr__(self) -> str:
        """
        Return a string representation of the Bates model.

        Returns
        -------
        str
            Model name + key parameters.
        """
        base = super().__repr__()
        return (
            f"{base}, kappa={self.kappa}, theta={self.theta}, sigma={self.sigma}, "
            f"rho={self.rho}, v0={self.v0}, jump_intensity={self.jump_intensity}, "
            f"muJ={self.muJ}, sigmaJ={self.sigmaJ}"
        )

    def __hashable_state__(self) -> tuple:
        """
        Generate a hashable state for caching or comparison.

        Returns
        -------
        tuple
            (kappa, theta, sigma, rho, v0, jump_intensity, muJ, sigmaJ)
        """
        return (
            self.kappa,
            self.theta,
            self.sigma,
            self.rho,
            self.v0,
            self.jump_intensity,
            self.muJ,
            self.sigmaJ,
        )
