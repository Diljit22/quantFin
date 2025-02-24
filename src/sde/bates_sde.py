#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
bates_sde.py
============
Implements SDE simulation for the Bates model, which extends Heston with Merton-like jumps.

The dynamics:
  - dS_t = S_t (r - q) dt + S_t sqrt(v_t) dW_1 + jumps
  - dv_t = kappa (theta - v_t) dt + sigma sqrt(v_t) dW_2
with correlation corr(dW_1, dW_2) = rho, plus Poisson jumps with intensity λ and
jump size ~ lognormal(μJ, σJ).

Methods
-------
- sample_paths
- sample_paths_and_derivative
"""

import math
import numpy as np
from typing import Optional, Tuple


class BatesSDE:
    """
    SDE simulator for the Bates model (Heston + Merton jumps).

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
    jump_intensity : float
        Poisson jump intensity (λ).
    muJ : float
        Mean jump size in log space.
    sigmaJ : float
        Vol of jump sizes in log space.
    random_state : int, optional
        Seed for RNG.
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
        random_state: Optional[int] = None,
    ) -> None:
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.v0 = v0
        self.jump_intensity = jump_intensity
        self.muJ = muJ
        self.sigmaJ = sigmaJ
        self._rng = np.random.default_rng(random_state)

    def sample_paths(
        self, T: float, n_sims: int, n_steps: int, r: float, q: float, S0: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate the Bates model using Euler approach for variance, plus jumps for price.

        Parameters
        ----------
        T : float
            Time to maturity.
        n_sims : int
            Number of paths.
        n_steps : int
            Time steps.
        r : float
            Risk-free rate.
        q : float
            Dividend yield.
        S0 : float
            Initial price.

        Returns
        -------
        (S, v)
            S shape (n_sims, n_steps+1) for the asset, v for variance.
        """
        dt = T / n_steps
        S = np.zeros((n_sims, n_steps + 1))
        v = np.zeros((n_sims, n_steps + 1))

        S[:, 0] = S0
        v[:, 0] = self.v0

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

            # SDE for price (like Heston):
            S_k = S[:, step]
            drift = (r - q) * S_k * dt
            vol_part = S_k * np.sqrt(v_k * dt)
            dS = drift + vol_part * W1
            S_next = np.maximum(S_k + dS, 1e-16)

            # Jumps:
            N_jumps = self._rng.poisson(self.jump_intensity * dt, size=n_sims)
            for i in range(n_sims):
                if N_jumps[i] > 0:
                    # sum the lognormal jumps
                    Y = self._rng.normal(self.muJ, self.sigmaJ, size=N_jumps[i])
                    S_next[i] *= math.exp(np.sum(Y))

            S[:, step + 1] = S_next
            v[:, step + 1] = v_next

        return S, v

    def sample_paths_and_derivative(
        self, T: float, n_sims: int, n_steps: int, r: float, q: float, S0: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate Bates paths (S, v) and compute ∂S/∂S0.

        Parameters
        ----------
        T : float
            Time to maturity.
        n_sims : int
            Number of paths.
        n_steps : int
            Time steps.
        r : float
            Risk-free rate.
        q : float
            Dividend yield.
        S0 : float
            Initial price.

        Returns
        -------
        (S, v, dSdS0)
            - S shape (n_sims, n_steps+1),
            - v shape (n_sims, n_steps+1),
            - dSdS0 shape (n_sims, n_steps+1).
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
            vol_part = S_k * np.sqrt(v_k * dt)
            S_next = S_k + drift + vol_part * W1
            S_next = np.maximum(S_next, 1e-16)

            ratio = S_next / (S_k + 1e-16)
            dS0_temp = dSdS0_k * ratio

            # Jumps
            N_jumps = self._rng.poisson(self.jump_intensity * dt, size=n_sims)
            for i in range(n_sims):
                if N_jumps[i] > 0:
                    Y = self._rng.normal(self.muJ, self.sigmaJ, size=N_jumps[i])
                    jump_factor = math.exp(np.sum(Y))
                    S_next[i] *= jump_factor
                    dS0_temp[i] *= jump_factor

            S[:, step + 1] = S_next
            v[:, step + 1] = v_next
            dSdS0[:, step + 1] = dS0_temp

        return S, v, dSdS0

    def __repr__(self) -> str:
        """
        String representation of BatesSDE.

        Returns
        -------
        str
            Representation including Heston + jump parameters.
        """
        return (
            f"{self.__class__.__name__}(kappa={self.kappa}, theta={self.theta}, "
            f"sigma={self.sigma}, rho={self.rho}, v0={self.v0}, "
            f"jump_intensity={self.jump_intensity}, muJ={self.muJ}, sigmaJ={self.sigmaJ})"
        )
