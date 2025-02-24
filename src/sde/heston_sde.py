#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
heston_sde.py
=============
Implements SDE simulation for the Heston model:

    dS_t = S_t (r - q) dt + S_t sqrt(v_t) dW_1
    dv_t = kappa (theta - v_t) dt + sigma sqrt(v_t) dW_2
with corr(dW_1, dW_2) = rho.

We use a basic Euler or Milstein approach. For partial derivative ∂S/∂S0,
we apply a ratio approach for the S-part. For v_t we do a straightforward
approximation.

Methods
-------
- sample_paths
- sample_paths_and_derivative
"""

import math
import numpy as np
from typing import Optional, Tuple


class HestonSDE:
    """
    SDE simulator for the Heston model.

    Parameters
    ----------
    kappa : float
        Mean-reversion speed for variance.
    theta : float
        Long-run variance.
    sigma : float
        Vol of vol.
    rho : float
        Correlation in [-1, 1].
    v0 : float
        Initial variance.
    random_state : Optional[int], optional
        Seed for random number generator.
    """

    def __init__(
        self,
        kappa: float,
        theta: float,
        sigma: float,
        rho: float,
        v0: float,
        random_state: Optional[int] = None,
    ):
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.v0 = v0
        self._rng = np.random.default_rng(random_state)

    def sample_paths(
        self, T: float, n_sims: int, n_steps: int, r: float, q: float, S0: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate asset price S and variance v under Heston using Euler approach.

        Parameters
        ----------
        T : float
            Time in years.
        n_sims : int
            Number of paths.
        n_steps : int
            Steps per path.
        r : float
            Risk-free rate.
        q : float
            Dividend yield.
        S0 : float
            Initial price.

        Returns
        -------
        (np.ndarray, np.ndarray)
            S of shape (n_sims, n_steps+1), v of same shape for variance.
        """
        dt = T / n_steps
        S = np.zeros((n_sims, n_steps + 1))
        v = np.zeros((n_sims, n_steps + 1))

        S[:, 0] = S0
        v[:, 0] = self.v0

        # Correlation approach: dW_1, dW_2 correlated => generate W_1, W'_2 and do W_2 = rho W_1 + sqrt(1-rho^2) W'_2
        for step in range(n_steps):
            Z1 = self._rng.normal(0, 1, size=n_sims)
            Z2 = self._rng.normal(0, 1, size=n_sims)
            W1 = Z1
            W2 = self.rho * Z1 + math.sqrt(1.0 - self.rho**2) * Z2

            # variance update
            v_k = np.maximum(v[:, step], 0.0)
            dv = (
                self.kappa * (self.theta - v_k) * dt
                + self.sigma * np.sqrt(v_k * dt) * W2
            )
            v_next = v_k + dv
            v_next = np.maximum(v_next, 0.0)  # ensure non-negative
            v[:, step + 1] = v_next

            # asset update
            S_k = S[:, step]
            drift = (r - q) * S_k * dt
            vol_part = np.sqrt(np.maximum(v_k, 0.0)) * S_k * math.sqrt(dt)
            dS = drift + vol_part * W1
            S_next = S_k + dS
            S[:, step + 1] = np.maximum(S_next, 1e-16)

        return S, v

    def sample_paths_and_derivative(
        self, T: float, n_sims: int, n_steps: int, r: float, q: float, S0: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate (S, v) under Heston and compute partial(S/∂S0).

        Parameters
        ----------
        T : float
            Time in years.
        n_sims : int
            Number of paths.
        n_steps : int
            Steps per path.
        r : float
            Risk-free rate.
        q : float
            Dividend yield.
        S0 : float
            Initial price.

        Returns
        -------
        (S, v, dSdS0)
            - S shape (n_sims, n_steps+1)
            - v shape (n_sims, n_steps+1)
            - dSdS0 shape (n_sims, n_steps+1)
        """
        dt = T / n_steps
        S = np.zeros((n_sims, n_steps + 1))
        v = np.zeros((n_sims, n_steps + 1))
        dSdS0 = np.zeros((n_sims, n_steps + 1))

        S[:, 0] = S0
        v[:, 0] = self.v0
        dSdS0[:, 0] = 1.0

        for step in range(n_steps):
            Z1 = self._rng.normal(0, 1, size=n_sims)
            Z2 = self._rng.normal(0, 1, size=n_sims)
            W1 = Z1
            W2 = self.rho * Z1 + math.sqrt(1.0 - self.rho**2) * Z2

            v_k = np.maximum(v[:, step], 0.0)
            dv = (
                self.kappa * (self.theta - v_k) * dt
                + self.sigma * np.sqrt(v_k * dt) * W2
            )
            v_next = v_k + dv
            v_next = np.maximum(v_next, 0.0)

            S_k = S[:, step]
            dSdS0_k = dSdS0[:, step]

            drift = (r - q) * S_k * dt
            vol_part = np.sqrt(v_k) * S_k * math.sqrt(dt)
            dS = drift + vol_part * W1
            S_next = S_k + dS
            ratio = (S_next + 1e-16) / (S_k + 1e-16)

            # partial(S_{k+1}, S_k) ~ 1 + partial(drift, S_k) + partial(vol_part, S_k)*W1
            # Here we'll do the ratio approach:
            dSdS0_next = dSdS0_k * ratio

            S_next = np.maximum(S_next, 1e-16)

            S[:, step + 1] = S_next
            v[:, step + 1] = v_next
            dSdS0[:, step + 1] = dSdS0_next

        return S, v, dSdS0

    def __repr__(self) -> str:
        """
        Return a string representation of the HestonSDE simulator.

        Returns
        -------
        str
            Representation including kappa, theta, sigma, rho, and v0.
        """
        return (
            f"{self.__class__.__name__}(kappa={self.kappa}, theta={self.theta}, "
            f"sigma={self.sigma}, rho={self.rho}, v0={self.v0})"
        )
