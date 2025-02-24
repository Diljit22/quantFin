#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
black_scholes_merton_sde.py
===========================
Implements an SDE simulation for the Black‐Scholes‐Merton model using a log-Euler discretization.

The simulation uses the dynamics:
    S_{k+1} = S_k * exp((r - q - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
where Z ~ N(0,1).

Usage:
    from sde.models.black_scholes_merton_sde import BlackScholesMertonSDE
    sde_sim = BlackScholesMertonSDE(sigma)
    paths = sde_sim.sample_paths(T, n_sims, n_steps)
"""

import math
import numpy as np
from typing import Tuple


class BlackScholesMertonSDE:
    """
    SDE simulator for the Black‐Scholes‐Merton model.

    Parameters
    ----------
    sigma : float
        The volatility parameter.
    """

    def __init__(self, sigma: float) -> None:
        self.sigma = sigma
        self._rng = np.random.default_rng()  # Numpy random generator

    def sample_paths(self, T: float, n_sims: int, n_steps: int) -> np.ndarray:
        """
        Simulate asset paths under BSM dynamics using the log-Euler method.

        Parameters
        ----------
        T : float
            Time to maturity in years.
        n_sims : int
            Number of simulation paths.
        n_steps : int
            Number of time steps.

        Returns
        -------
        np.ndarray
            A 2D array of simulated asset paths of shape (n_sims, n_steps+1).
        """
        dt = T / n_steps
        S = np.zeros((n_sims, n_steps + 1), dtype=float)
        # Assume an initial price S0=1.0; the actual initial value can be scaled externally.
        S[:, 0] = 1.0
        drift = -0.5 * self.sigma**2 * dt
        vol = self.sigma * math.sqrt(dt)
        for step in range(n_steps):
            Z = self._rng.normal(0, 1, size=n_sims)
            S[:, step + 1] = S[:, step] * np.exp(drift + vol * Z)
        return S

    def sample_paths_and_derivative(
        self, T: float, n_sims: int, n_steps: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate asset paths and compute their sensitivity with respect to the initial price S0.

        Parameters
        ----------
        T : float
            Time to maturity in years.
        n_sims : int
            Number of simulation paths.
        n_steps : int
            Number of time steps.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple (S, dSdS0) where:
              - S is the array of simulated asset paths.
              - dSdS0 is the sensitivity (derivative) of S with respect to S0.
        """
        dt = T / n_steps
        S = np.zeros((n_sims, n_steps + 1), dtype=float)
        dSdS0 = np.zeros((n_sims, n_steps + 1), dtype=float)
        S[:, 0] = 1.0
        dSdS0[:, 0] = 1.0
        drift = -0.5 * self.sigma**2 * dt
        vol = self.sigma * math.sqrt(dt)
        for step in range(n_steps):
            Z = self._rng.normal(0, 1, size=n_sims)
            S_next = S[:, step] * np.exp(drift + vol * Z)
            ratio = S_next / (S[:, step] + 1e-16)
            S[:, step + 1] = S_next
            dSdS0[:, step + 1] = dSdS0[:, step] * ratio
        return S, dSdS0

    def __repr__(self) -> str:
        """
        Return a string representation of the SDE simulator.

        Returns
        -------
        str
            A representation including the volatility parameter.
        """
        return f"{self.__class__.__name__}(sigma={self.sigma})"
