#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
kou_sde.py
==========
Simulates the Kou double-exponential jump-diffusion model via an Euler or log-Euler scheme.

Dynamics:
  dS_t = S_t (r - q) dt + sigma S_t dW_t + jumps,
where jumps come from a Poisson(λ dt) distribution, each jump drawn from a mixture
of exponentials: prob p => +Exp(alpha1), else => -Exp(alpha2).

Methods
-------
- sample_paths
- sample_paths_and_derivative
"""

import math
import numpy as np
from typing import Optional, Tuple


class KouSDE:
    """
    SDE simulator for the Kou jump-diffusion model.

    Parameters
    ----------
    sigma : float
        Diffusive volatility.
    jump_intensity : float
        Poisson jump intensity (λ).
    p_up : float
        Probability of an upward jump.
    alpha1 : float
        Rate for upward jumps.
    alpha2 : float
        Rate for downward jumps.
    random_state : int, optional
        Seed for RNG.
    """

    def __init__(
        self,
        sigma: float,
        jump_intensity: float,
        p_up: float,
        alpha1: float,
        alpha2: float,
        random_state: Optional[int] = None,
    ) -> None:
        self.sigma = sigma
        self.jump_intensity = jump_intensity
        self.p_up = p_up
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self._rng = np.random.default_rng(random_state)

    def sample_paths(
        self, T: float, n_sims: int, n_steps: int, r: float, q: float, S0: float
    ) -> np.ndarray:
        """
        Simulate paths under Kou using a log-Euler approach for the continuous part, plus Poisson jumps.

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
        np.ndarray
            A 2D array of shape (n_sims, n_steps+1).
        """
        dt = T / n_steps
        S = np.zeros((n_sims, n_steps + 1), dtype=float)
        S[:, 0] = S0

        drift = (r - q - 0.5 * self.sigma * self.sigma) * dt
        vol = self.sigma * math.sqrt(dt)

        for step in range(n_steps):
            Z = self._rng.normal(0, 1, size=n_sims)
            log_s_k = np.log(S[:, step] + 1e-16)
            log_s_next = log_s_k + drift + vol * Z
            S_next = np.exp(log_s_next)
            # Poisson jumps
            N_jumps = self._rng.poisson(self.jump_intensity * dt, size=n_sims)
            for i in range(n_sims):
                if N_jumps[i] > 0:
                    # sum of N_jumps[i] many jumps. Each jump is up or down:
                    jump_total = 0.0
                    for _ in range(N_jumps[i]):
                        if self._rng.random() < self.p_up:
                            # upward jump from Exp(alpha1)
                            jump_size = self._rng.exponential(1.0 / self.alpha1)
                            S_next[i] *= math.exp(jump_size)
                        else:
                            # downward jump
                            jump_size = self._rng.exponential(1.0 / self.alpha2)
                            S_next[i] *= math.exp(-jump_size)
            S[:, step + 1] = S_next
        return S

    def sample_paths_and_derivative(
        self, T: float, n_sims: int, n_steps: int, r: float, q: float, S0: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate paths under Kou and compute partial derivative w.r.t. S0.

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
        (S, dSdS0)
            Both shape (n_sims, n_steps+1).
        """
        dt = T / n_steps
        S = np.zeros((n_sims, n_steps + 1), dtype=float)
        dSdS0 = np.zeros((n_sims, n_steps + 1), dtype=float)

        S[:, 0] = S0
        dSdS0[:, 0] = 1.0

        drift = (r - q - 0.5 * self.sigma * self.sigma) * dt
        vol = self.sigma * math.sqrt(dt)

        for step in range(n_steps):
            Z = self._rng.normal(0, 1, size=n_sims)
            log_s_k = np.log(S[:, step] + 1e-16)
            log_s_next = log_s_k + drift + vol * Z
            S_next = np.exp(log_s_next)

            ratio = S_next / (S[:, step] + 1e-16)
            dSdS0_temp = dSdS0[:, step] * ratio

            # jumps
            N_jumps = self._rng.poisson(self.jump_intensity * dt, size=n_sims)
            for i in range(n_sims):
                if N_jumps[i] > 0:
                    for _ in range(N_jumps[i]):
                        if self._rng.random() < self.p_up:
                            jsize = self._rng.exponential(1.0 / self.alpha1)
                            factor = math.exp(jsize)
                            S_next[i] *= factor
                            dSdS0_temp[i] *= factor
                        else:
                            jsize = self._rng.exponential(1.0 / self.alpha2)
                            factor = math.exp(-jsize)
                            S_next[i] *= factor
                            dSdS0_temp[i] *= factor

            S[:, step + 1] = S_next
            dSdS0[:, step + 1] = dSdS0_temp

        return S, dSdS0

    def __repr__(self) -> str:
        """
        String representation for debugging.

        Returns
        -------
        str
            A representation with sigma, jump_intensity, p_up, alpha1, alpha2.
        """
        return (
            f"{self.__class__.__name__}(sigma={self.sigma}, jump_intensity={self.jump_intensity}, "
            f"p_up={self.p_up}, alpha1={self.alpha1}, alpha2={self.alpha2})"
        )
