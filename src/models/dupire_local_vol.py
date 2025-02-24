#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dupire_local_vol.py
==================
Unified implementation of the Dupire Local Volatility model.

This model uses a local volatility function σ(S, t) to price options.
It supports:
  - SDE simulation via DupireLocalVolSDE.
  - (Optional) Closed-form pricing via numerical integration (not implemented).
  - Calibration support through update_params() if needed.

Market data parameters (r, q, S0) are provided externally during pricing.
The intrinsic parameter is the local volatility function.
"""

from typing import Callable, Dict, Any
import math
from src.models.base_model import BaseModel
from src.sde.dupire_local_vol_sde import DupireLocalVolSDE


class DupireLocalVol(BaseModel):
    """
    Dupire Local Volatility model for option pricing.

    The model's intrinsic parameter is the local volatility function,
    which should be provided as a callable: σ = local_vol_func(S, t).

    Note:
    A closed-form characteristic function and pricing solution are not generally available.
    """

    def __init__(self, local_vol_func: Callable[[float, float], float]) -> None:
        """
        Initialize the Dupire Local Volatility model.

        Parameters
        ----------
        local_vol_func : callable
            A function that takes (S, t) and returns the local volatility σ(S, t).
        """
        super().__init__(model_name="DupireLocalVol", local_vol_func=local_vol_func)

    @property
    def local_vol_func(self) -> Callable[[float, float], float]:
        """
        callable : Returns the local volatility function.
        """
        return self._params["local_vol_func"]

    def validate_params(self) -> None:
        """
        Validate the model parameters.

        Raises
        ------
        ValueError
            If the local volatility function is not callable.
        """
        if not callable(self.local_vol_func):
            raise ValueError("local_vol_func must be callable.")

    def SDE(self) -> DupireLocalVolSDE:
        """
        Return an instance of the DupireLocalVolSDE simulator.

        Returns
        -------
        DupireLocalVolSDE
            The SDE simulator configured with the local volatility function.
        """
        return DupireLocalVolSDE(self.local_vol_func)

    def characteristic_function(self, *args, **kwargs):
        """
        Local volatility models are generally non-affine, and a closed-form characteristic
        function is not available.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError(
            "Characteristic function is not available for Dupire Local Volatility."
        )

    def price_option(self, S: float, K: float, T: float, r: float, q: float) -> float:
        """
        Price an option using a closed-form solution or numerical integration approach.

        Parameters
        ----------
        S : float
            Underlying asset price.
        K : float
            Strike price.
        T : float
            Time to maturity.
        r : float
            Risk-free rate.
        q : float
            Dividend yield.

        Returns
        -------
        float
            Option price.

        Raises
        ------
        NotImplementedError
            Because closed-form pricing is not implemented.
        """
        raise NotImplementedError(
            "Closed-form pricing for Dupire Local Volatility is not implemented."
        )

    def update_params(self, new_params: Dict[str, Any]) -> None:
        """
        Update the model parameters with new values.

        Parameters
        ----------
        new_params : dict
            A dictionary of new parameter values.
        """
        self._params.update(new_params)

    def __repr__(self) -> str:
        """
        Return a string representation of the DupireLocalVol model.

        Returns
        -------
        str
            A representation including the model name and the local volatility function.
        """
        func_name = (
            self.local_vol_func.__name__
            if hasattr(self.local_vol_func, "__name__")
            else str(self.local_vol_func)
        )
        base = super().__repr__()
        return f"{base}, local_vol_func={func_name}"

    def __hashable_state__(self) -> tuple:
        """
        Generate a hashable state for caching or comparison.

        Returns
        -------
        tuple
            A tuple identifying the local volatility function.
        """
        func_name = (
            self.local_vol_func.__name__
            if hasattr(self.local_vol_func, "__name__")
            else str(self.local_vol_func)
        )
        return (func_name,)
