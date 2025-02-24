#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
merton_jump.py
==============
Unified implementation of the Merton jump-diffusion model.

This model extends the Black-Scholes-Merton framework by incorporating jumps.
It supports:
  - Calculation of the characteristic function (via a standalone function).
  - Simulation via an SDE (using MertonJumpSDE).
  - Option pricing via a closed-form method (implemented externally).
  - Calibration support through update_params() and price_option() methods.

Market data parameters (r, q, S_0) are provided externally during pricing.
Intrinsic model parameters (sigma, jump_intensity, muJ, sigmaJ) are stored in the model.
"""

from typing import Callable, Dict, Any

from src.models.base_model import BaseModel
from src.characteristic_equations.merton_jump_cf import merton_jump_cf
from src.sde.merton_jump_sde import MertonJumpSDE


class MertonJump(BaseModel):
    """
    Merton jump-diffusion model for option pricing.

    Attributes
    ----------
    _params["sigma"] : float
        Diffusive volatility.
    _params["jump_intensity"] : float
        Jump intensity (λ).
    _params["muJ"] : float
        Mean jump size (log-scale).
    _params["sigmaJ"] : float
        Volatility of jump sizes.
    """

    def __init__(
        self, sigma: float, jump_intensity: float, muJ: float, sigmaJ: float
    ) -> None:
        """
        Initialize the Merton jump-diffusion model.

        Parameters
        ----------
        sigma : float
            Diffusive volatility.
        jump_intensity : float
            Jump intensity (λ).
        muJ : float
            Mean jump size (log-scale).
        sigmaJ : float
            Volatility of jump sizes.
        """
        super().__init__(
            model_name="MertonJump",
            sigma=sigma,
            jump_intensity=jump_intensity,
            muJ=muJ,
            sigmaJ=sigmaJ,
        )

    @property
    def sigma(self) -> float:
        """Return the diffusive volatility."""
        return self._params["sigma"]

    @property
    def jump_intensity(self) -> float:
        """Return the jump intensity (λ)."""
        return self._params["jump_intensity"]

    @property
    def muJ(self) -> float:
        """Return the mean jump size (log-scale)."""
        return self._params["muJ"]

    @property
    def sigmaJ(self) -> float:
        """Return the volatility of jump sizes."""
        return self._params["sigmaJ"]

    def validate_params(self) -> None:
        """
        Validate the model parameters.

        Raises
        ------
        ValueError
            If any required parameter is invalid.
        """
        if self.sigma < 0:
            raise ValueError("Diffusive volatility (sigma) must be non-negative.")
        # Additional validations for jump parameters can be added as needed.

    def characteristic_function(
        self, t: float, spot: float, r: float, q: float
    ) -> Callable[[complex], complex]:
        """
        Return the characteristic function for the Merton jump-diffusion model.

        Parameters
        ----------
        t : float
            Time to maturity in years.
        spot : float
            Current asset spot price.
        r : float
            Risk-free rate.
        q : float
            Dividend yield.

        Returns
        -------
        Callable[[complex], complex]
            A function φ(u) that computes the characteristic function value.
        """
        return merton_jump_cf(
            t, spot, r, q, self.sigma, self.jump_intensity, self.muJ, self.sigmaJ
        )

    def SDE(self) -> MertonJumpSDE:
        """
        Return an instance of the SDE simulator for the Merton jump-diffusion model.

        Returns
        -------
        MertonJumpSDE
            The SDE simulator configured with the model's intrinsic parameters.
        """
        return MertonJumpSDE(self.sigma, self.jump_intensity, self.muJ, self.sigmaJ)

    def price_option(self, S: float, K: float, T: float, r: float, q: float) -> float:
        """
        Price an option using a closed-form solution for the Merton jump-diffusion model.

        This method should compute the theoretical price of an option based on the model.
        (The actual closed-form implementation is to be developed later.)

        Parameters
        ----------
        S : float
            Underlying asset price.
        K : float
            Strike price.
        T : float
            Time to maturity in years.
        r : float
            Risk-free rate.
        q : float
            Dividend yield.

        Returns
        -------
        float
            Theoretical option price.

        Notes
        -----
        Calibration routines will call this method.
        """
        raise NotImplementedError(
            "Closed-form pricing for MertonJump is not implemented yet."
        )

    def update_params(self, new_params: Dict[str, Any]) -> None:
        """
        Update the model parameters with new values.

        Parameters
        ----------
        new_params : dict
            A dictionary containing new parameter values.
        """
        self._params.update(new_params)

    def __repr__(self) -> str:
        """
        Return a string representation of the MertonJump model.

        Returns
        -------
        str
            A representation including the model name and jump parameters.
        """
        base = super().__repr__()
        return f"{base}, jump_intensity={self.jump_intensity}, muJ={self.muJ}, sigmaJ={self.sigmaJ}"

    def __hashable_state__(self) -> tuple:
        """
        Generate a hashable state for caching or comparison.

        Returns
        -------
        tuple
            A tuple containing intrinsic model parameters.
        """
        return (self.sigma, self.jump_intensity, self.muJ, self.sigmaJ)
