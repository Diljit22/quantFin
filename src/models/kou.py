#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
kou.py
======
Defines the Kou jump-diffusion model as a class inheriting from BaseModel.

Parameters in _params:
  - sigma : float
  - jump_intensity : float
  - p_up : float
  - alpha1 : float
  - alpha2 : float

No closed-form pricing is provided here, though partial results exist in the literature.
"""

from typing import Callable, Dict, Any
from src.models.base_model import BaseModel
from src.characteristic_equations.kou_cf import kou_cf
from src.sde.kou_sde import KouSDE


class Kou(BaseModel):
    """
    Kou jump-diffusion model for option pricing.

    Attributes
    ----------
    _params : dict
        - sigma : diffusive volatility
        - jump_intensity : Poisson jump intensity
        - p_up : probability of upward jump
        - alpha1 : rate for upward jump
        - alpha2 : rate for downward jump
    """

    def __init__(
        self,
        sigma: float,
        jump_intensity: float,
        p_up: float,
        alpha1: float,
        alpha2: float,
    ) -> None:
        """
        Initialize the Kou model with double-exponential jumps.

        Parameters
        ----------
        sigma : float
            Diffusive volatility.
        jump_intensity : float
            Poisson jump intensity (lambda).
        p_up : float
            Probability of an upward jump.
        alpha1 : float
            Rate for upward exponential jumps.
        alpha2 : float
            Rate for downward exponential jumps.
        """
        super().__init__(
            model_name="Kou",
            sigma=sigma,
            jump_intensity=jump_intensity,
            p_up=p_up,
            alpha1=alpha1,
            alpha2=alpha2,
        )

    @property
    def sigma(self) -> float:
        """Diffusive volatility."""
        return self._params["sigma"]

    @property
    def jump_intensity(self) -> float:
        """Poisson jump intensity (lambda)."""
        return self._params["jump_intensity"]

    @property
    def p_up(self) -> float:
        """Probability of an upward jump in the mixture."""
        return self._params["p_up"]

    @property
    def alpha1(self) -> float:
        """Rate for upward jumps (Exp)."""
        return self._params["alpha1"]

    @property
    def alpha2(self) -> float:
        """Rate for downward jumps (Exp)."""
        return self._params["alpha2"]

    def validate_params(self) -> None:
        """
        Validate the model parameters.

        Raises
        ------
        ValueError
            If parameter domain is invalid.
        """
        if self.sigma < 0.0:
            raise ValueError("sigma must be >= 0.")
        if not 0.0 <= self.p_up <= 1.0:
            raise ValueError("p_up must be in [0,1].")
        if self.alpha1 <= 0.0 or self.alpha2 <= 0.0:
            raise ValueError("alpha1, alpha2 must be > 0.")
        if self.jump_intensity < 0.0:
            raise ValueError("jump_intensity must be >= 0.")

    def characteristic_function(
        self, t: float, spot: float, r: float, q: float
    ) -> Callable[[complex], complex]:
        """
        Return the characteristic function of the Kou model's log-price.

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
            phi(u) that computes the CF at complex u.
        """
        return kou_cf(
            t,
            spot,
            r,
            q,
            self.sigma,
            self.jump_intensity,
            self.p_up,
            self.alpha1,
            self.alpha2,
        )

    def SDE(self) -> KouSDE:
        """
        Return an instance of the KouSDE simulator.

        Returns
        -------
        KouSDE
            SDE simulator for Kou's model.
        """
        return KouSDE(
            sigma=self.sigma,
            jump_intensity=self.jump_intensity,
            p_up=self.p_up,
            alpha1=self.alpha1,
            alpha2=self.alpha2,
        )

    def price_option(self, *args, **kwargs) -> float:
        """
        Placeholder for option pricing under Kou's model.

        Raises
        ------
        NotImplementedError
            Not implemented.
        """
        raise NotImplementedError("Closed-form pricing for Kou is not implemented.")

    def update_params(self, new_params: Dict[str, Any]) -> None:
        """
        Update the model parameters with new values.

        Parameters
        ----------
        new_params : dict
            Potentially includes keys: 'sigma','jump_intensity','p_up','alpha1','alpha2'.
        """
        self._params.update(new_params)

    def __repr__(self) -> str:
        """
        Return a string representation of the Kou model.

        Returns
        -------
        str
            Name and key parameters.
        """
        base = super().__repr__()
        return (
            f"{base}, jump_intensity={self.jump_intensity}, p_up={self.p_up}, "
            f"alpha1={self.alpha1}, alpha2={self.alpha2}"
        )

    def __hashable_state__(self) -> tuple:
        """
        Generate a hashable state for caching or comparison.

        Returns
        -------
        tuple
            (sigma, jump_intensity, p_up, alpha1, alpha2).
        """
        return (self.sigma, self.jump_intensity, self.p_up, self.alpha1, self.alpha2)
