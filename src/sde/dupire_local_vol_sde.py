#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dupire_local_vol_sde.py
======================
SDE simulation for the Dupire Local Volatility model.

The local volatility function is a callable σ(S, t).
Asset dynamics:
    dS_t = (r - q) * S_t dt + σ(S_t, t) * S_t dW_t.

Methods
-------
- sample_paths
- sample_paths_and_derivative
"""

import math
import numpy as np
from typing import Callable, Optional, Tuple
from numpy.random import default_rng


class DupireLocalVolSDE:
    """
    SDE simulator for the Dupire Local Volatility model.

    Parameters
    ----------
    local_vol_func : Callable[[float, float], float]
        A function that takes the current asset price S and time t,
        and returns the local volatility σ(S, t).
    random_state : int, optional
        Seed for the random number generator (default is None).
    """

    def __init__(
        self,
        local_vol_func: Callable[[float, float], float],
        random_state: Optional[int] = None,
    ) -> None:
        self.local_vol_func = local_vol_func
        self._rng = default_rng(random_state)

    def sample_paths(
        self, T: float, n_sims: int, n_steps: int, r: float, q: float, S0: float
    ) -> np.ndarray:
        """
        Simulate asset paths under the Dupire Local Vol model using a log-Euler approach.

        Parameters
        ----------
        T : float
            Time to maturity.
        n_sims : int
            Number of simulation paths.
        n_steps : int
            Number of time steps.
        r : float
            Risk-free interest rate.
        q : float
            Dividend yield.
        S0 : float
            Initial asset price.

        Returns
        -------
        np.ndarray
            A 2D array of shape (n_sims, n_steps+1).
        """
        dt = T / n_steps
        S = np.zeros((n_sims, n_steps + 1), dtype=float)
        S[:, 0] = S0

        for step in range(n_steps):
            t = step * dt
            # local vol for each sim at current S
            sigma_local = np.array([self.local_vol_func(s, t) for s in S[:, step]])
            drift = (r - q - 0.5 * sigma_local**2) * dt
            vol = sigma_local * math.sqrt(dt)

            Z = self._rng.normal(0, 1, size=n_sims)
            log_s_k = np.log(S[:, step] + 1e-16)
            log_s_next = log_s_k + drift + vol * Z
            S[:, step + 1] = np.exp(log_s_next)

        return S

    def sample_paths_and_derivative(
        self, T: float, n_sims: int, n_steps: int, r: float, q: float, S0: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate paths and compute ∂S/∂S0 for Dupire Local Vol.

        Parameters
        ----------
        T : float
            Time to maturity.
        n_sims : int
            Number of simulation paths.
        n_steps : int
            Number of time steps.
        r : float
            Risk-free interest rate.
        q : float
            Dividend yield.
        S0 : float
            Initial asset price.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            S, dSdS0: both shape (n_sims, n_steps+1).
        """
        dt = T / n_steps
        S = np.zeros((n_sims, n_steps + 1), dtype=float)
        dSdS0 = np.zeros((n_sims, n_steps + 1), dtype=float)

        S[:, 0] = S0
        dSdS0[:, 0] = 1.0

        for step in range(n_steps):
            t = step * dt
            sigma_local = np.array([self.local_vol_func(s, t) for s in S[:, step]])
            drift = (r - q - 0.5 * sigma_local**2) * dt
            vol = sigma_local * math.sqrt(dt)

            Z = self._rng.normal(0, 1, size=n_sims)
            log_s_k = np.log(S[:, step] + 1e-16)
            log_s_next = log_s_k + drift + vol * Z
            S_next = np.exp(log_s_next)

            ratio = S_next / (S[:, step] + 1e-16)
            dS_temp = dSdS0[:, step] * ratio

            S[:, step + 1] = S_next
            dSdS0[:, step + 1] = dS_temp

        return S, dSdS0

    def __repr__(self) -> str:
        """
        Return a string representation of the DupireLocalVolSDE simulator.

        Returns
        -------
        str
            Name of the local volatility function if possible.
        """
        func_name = (
            self.local_vol_func.__name__
            if hasattr(self.local_vol_func, "__name__")
            else str(self.local_vol_func)
        )
        return f"{self.__class__.__name__}(local_vol_func={func_name})"
