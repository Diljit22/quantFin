#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
variance_gamma_sde.py
=====================
Simulates the Variance Gamma process via a time-change approach:

  X_{t+dt} - X_t = theta * G_{dt} + sigma * sqrt(G_{dt}) * Z,
where G_{dt} is a Gamma random variable with shape = dt/nu, scale=nu,
and Z ~ N(0,1).

We exponentiate for the asset price: S_{k+1} = S_k * exp(Delta X).

Methods
-------
- sample_paths
- sample_paths_and_derivative
"""

import math
import numpy as np
from typing import Optional, Tuple


class VarianceGammaSDE:
    """
    SDE-like simulator for the Variance Gamma model.

    Parameters
    ----------
    sigma : float
        Vol parameter in the Brownian part.
    theta : float
        Drift in the Brownian part.
    nu : float
        Gamma scale (1/nu is shape param).
    random_state : int, optional
        Seed for RNG.
    """

    def __init__(
        self, sigma: float, theta: float, nu: float, random_state: Optional[int] = None
    ) -> None:
        self.sigma = sigma
        self.theta = theta
        self.nu = nu
        self._rng = np.random.default_rng(random_state)

    def sample_paths(
        self, T: float, n_sims: int, n_steps: int, r: float, q: float, S0: float
    ) -> np.ndarray:
        """
        Simulate paths for the VG process + risk-neutral correction.
        We do a multi-step approach:

          For each step dt = T / n_steps:
            G ~ Gamma(shape=dt/nu, scale=nu),
            Z ~ N(0,1),
            dX = theta * G + sigma * sqrt(G) * Z,
            S_{k+1} = S_k * exp(dX + drift dt?), or we incorporate the drift correction
                      directly in the CF approach. We'll keep an extra (r-q)? => or do it in a step.

        For simplicity, we do no continuous compounding from (r-q) in this snippet; we rely on
        the model's characteristic function approach to ensure risk-neutral. But if we want
        an explicit e^{(r-q)*dt}, we can add that. We'll keep the code commented for that.

        Parameters
        ----------
        T : float
            Time in years.
        n_sims : int
            Number of paths.
        n_steps : int
            Steps.
        r : float
            Risk-free rate (often omitted if drift is purely from CF correction).
        q : float
            Dividend yield (same note as r).
        S0 : float
            Initial price.

        Returns
        -------
        np.ndarray
            shape (n_sims, n_steps+1)
        """
        dt = T / n_steps
        S = np.zeros((n_sims, n_steps + 1), dtype=float)
        S[:, 0] = S0

        for step in range(n_steps):
            # sample gamma
            shape_param = dt / self.nu
            scale_param = self.nu
            G = self._rng.gamma(shape=shape_param, scale=scale_param, size=n_sims)
            # sample normal
            Z = self._rng.normal(0, 1, size=n_sims)

            dX = self.theta * G + self.sigma * np.sqrt(G) * Z
            # optional + (r-q)*dt if we want direct e^{(r-q)*dt} in each step.
            # We'll omit it if we rely on a CF drift correction at a model level.
            # dX += (r-q)*dt ?

            incr = np.exp(dX)
            S[:, step + 1] = S[:, step] * incr

        return S

    def sample_paths_and_derivative(
        self, T: float, n_sims: int, n_steps: int, r: float, q: float, S0: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate VG paths and partial derivative w.r.t S0 using ratio approach.

        Parameters
        ----------
        T : float
            Maturity.
        n_sims : int
            # of paths.
        n_steps : int
            steps.
        r : float
            Risk-free rate or 0 if included in drift correction.
        q : float
            Dividend yield or 0 if drift corrected.
        S0 : float
            initial price.

        Returns
        -------
        (S, dSdS0)
            shapes (n_sims, n_steps+1).
        """
        dt = T / n_steps
        S = np.zeros((n_sims, n_steps + 1), dtype=float)
        dSdS0 = np.zeros((n_sims, n_steps + 1), dtype=float)

        S[:, 0] = S0
        dSdS0[:, 0] = 1.0

        for step in range(n_steps):
            # gamma subordinator
            shape_param = dt / self.nu
            scale_param = self.nu
            G = self._rng.gamma(shape=shape_param, scale=scale_param, size=n_sims)

            Z = self._rng.normal(0, 1, size=n_sims)
            dX = self.theta * G + self.sigma * np.sqrt(G) * Z
            incr = np.exp(dX)

            S_next = S[:, step] * incr
            ratio = S_next / (S[:, step] + 1e-16)
            dSdS0_next = dSdS0[:, step] * ratio

            S[:, step + 1] = S_next
            dSdS0[:, step + 1] = dSdS0_next

        return S, dSdS0

    def __repr__(self) -> str:
        """
        Return a string representation of the VG SDE.

        Returns
        -------
        str
        """
        return (
            f"{self.__class__.__name__}(sigma={self.sigma}, theta={self.theta}, "
            f"nu={self.nu})"
        )
