#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
nig.py
======
Defines the NIG model class, referencing the CF (nig_cf) and an approximate SDE (nig_sde).

Parameters: (alpha, beta, delta). We incorporate a drift correction in the CF
so that E[S_t] = S_0 e^{(r-q) t}.

Usage
-----
    from src.models.nig import NIG
    model = NIG(alpha=10.0, beta=-2.0, delta=0.3)
    cf = model.characteristic_function(t=1.0, spot=100, r=0.05, q=0.02)
    sde = model.SDE()
    paths = sde.sample_paths(T=1.0, n_sims=10000, n_steps=100, r=0.05, q=0.02, S0=100)
"""

from typing import Callable, Dict, Any
from src.models.base_model import BaseModel
from src.characteristic_equations.nig_cf import nig_cf
from src.sde.nig_sde import NIGSDE

class NIG(BaseModel):
    """
    Normal Inverse Gaussian model for log-price increments.

    The CF includes drift correction. The SDE is approximate.

    Attributes
    ----------
    alpha, beta, delta
    """

    def __init__(self, alpha: float, beta: float, delta: float) -> None:
        """
        Initialize the NIG model.

        Parameters
        ----------
        alpha : float
        beta : float
        delta : float
        """
        super().__init__(
            model_name="NIG",
            alpha=alpha,
            beta=beta,
            delta=delta
        )

    @property
    def alpha(self) -> float:
        """NIG alpha > 0."""
        return self._params["alpha"]

    @property
    def beta(self) -> float:
        """NIG beta with |beta| < alpha."""
        return self._params["beta"]

    @property
    def delta(self) -> float:
        """NIG scale delta > 0."""
        return self._params["delta"]

    def validate_params(self) -> None:
        """
        Validate domain constraints.

        Raises
        ------
        ValueError
            If invalid domain.
        """
        if self.alpha <= 0.0:
            raise ValueError("alpha must be > 0.")
        if abs(self.beta) >= self.alpha:
            raise ValueError("Must have |beta| < alpha.")
        if self.delta <= 0.0:
            raise ValueError("delta must be > 0.")

    def characteristic_function(
        self, t: float, spot: float, r: float, q: float
    ) -> Callable[[complex], complex]:
        """
        Return the NIG characteristic function for ln(S_t).

        Parameters
        ----------
        t : float
        spot : float
        r : float
        q : float

        Returns
        -------
        Callable[[complex], complex]
        """
        return nig_cf(
            t, spot, r, q,
            self.alpha,
            self.beta,
            self.delta
        )

    def SDE(self) -> NIGSDE:
        """
        Return an approximate SDE simulator.

        Returns
        -------
        NIGSDE
        """
        return NIGSDE(
            alpha=self.alpha,
            beta=self.beta,
            delta=self.delta
        )

    def price_option(self, *args, **kwargs) -> float:
        """
        Placeholder for numeric integral or expansions.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError("Use numeric integral approach for NIG pricing.")

    def update_params(self, new_params: Dict[str, Any]) -> None:
        """
        Update parameters.

        Parameters
        ----------
        new_params : dict
        """
        self._params.update(new_params)

    def __repr__(self) -> str:
        base = super().__repr__()
        return f"{base}, alpha={self.alpha}, beta={self.beta}, delta={self.delta}"

    def __hashable_state__(self) -> tuple:
        return (self.alpha, self.beta, self.delta)
