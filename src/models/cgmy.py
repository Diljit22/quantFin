#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cgmy.py
=======
Defines the CGMY model class, referencing the CF (cgmy_cf) and an approximate SDE (cgmy_sde).

We have parameters (C, G, M, Y). The risk-neutral drift correction is embedded
in the CF. The SDE approach is approximate for large jumps and small jumps.

Usage
-----
    from src.models.cgmy import CGMY
    model = CGMY(C=1.0, G=5.0, M=5.0, Y=0.5)
    cf = model.characteristic_function(t=1.0, spot=100, r=0.05, q=0.02)
    sde = model.SDE()
    paths = sde.sample_paths(...)
"""

from typing import Callable, Dict, Any
from src.models.base_model import BaseModel
from src.characteristic_equations.cgmy_cf import cgmy_cf
from src.sde.cgmy_sde import CGMYSDE

class CGMY(BaseModel):
    """
    CGMY model for pure jump processes. Four parameters:
      - C : scale
      - G : positive rate
      - M : positive rate
      - Y : fractional exponent in (0,2)

    The CF includes the drift correction so that E[S_t] = S_0 e^{(r-q) t}.
    The SDE is an approximate scheme for simulating paths.
    """

    def __init__(
        self,
        C: float,
        G: float,
        M: float,
        Y: float
    ) -> None:
        """
        Initialize the CGMY model.

        Parameters
        ----------
        C : float
            Scale param, >0
        G : float
            Usually >0
        M : float
            Usually >0
        Y : float
            In (0,2)
        """
        super().__init__(
            model_name="CGMY",
            C=C,
            G=G,
            M=M,
            Y=Y
        )

    @property
    def C(self) -> float:
        """C param."""
        return self._params["C"]

    @property
    def G(self) -> float:
        """G param."""
        return self._params["G"]

    @property
    def M(self) -> float:
        """M param."""
        return self._params["M"]

    @property
    def Y(self) -> float:
        """Y param in (0,2)."""
        return self._params["Y"]

    def validate_params(self) -> None:
        """
        Validate CGMY parameters.

        Raises
        ------
        ValueError
            If domain constraints are violated.
        """
        if self.C <= 0.0:
            raise ValueError("C must be > 0.")
        if self.G <= 0.0 or self.M <= 0.0:
            raise ValueError("G, M must be > 0.")
        if not (0.0 < self.Y < 2.0):
            raise ValueError("Y must be in (0,2).")

    def characteristic_function(
        self, t: float, spot: float, r: float, q: float
    ) -> Callable[[complex], complex]:
        """
        Return the CGMY characteristic function for ln(S_t).

        Parameters
        ----------
        t : float
            Time in years.
        spot : float
            Spot price S0.
        r : float
            Risk-free rate.
        q : float
            Dividend yield.

        Returns
        -------
        Callable[[complex], complex]
            phi(u).
        """
        from src.characteristic_equations.cgmy_cf import cgmy_cf
        return cgmy_cf(
            t, spot, r, q,
            self.C, self.G, self.M, self.Y
        )

    def SDE(self) -> CGMYSDE:
        """
        Return an approximate SDE simulator for CGMY.

        Returns
        -------
        CGMYSDE
        """
        return CGMYSDE(
            C=self.C,
            G=self.G,
            M=self.M,
            Y=self.Y
        )

    def price_option(self, *args, **kwargs) -> float:
        """
        Placeholder for numeric integral approach to call pricing.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError("Use a separate numeric integral or expansions for CGMY pricing.")

    def update_params(self, new_params: Dict[str, Any]) -> None:
        """
        Update CGMY parameters.

        Parameters
        ----------
        new_params : dict
            Possibly including 'C','G','M','Y'.
        """
        self._params.update(new_params)

    def __repr__(self) -> str:
        """
        Debugging representation.

        Returns
        -------
        str
        """
        base = super().__repr__()
        return (f"{base}, C={self.C}, G={self.G}, M={self.M}, Y={self.Y}")

    def __hashable_state__(self) -> tuple:
        """
        Hashable state.

        Returns
        -------
        tuple
        """
        return (self.C, self.G, self.M, self.Y)
