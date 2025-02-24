#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
merton_jump_sde.py
==================
SDE simulation for the Merton jump-diffusion model.

Under the Merton jump-diffusion framework, asset dynamics are:
    S_{t+dt} = S_t * exp((r - q - λ(E[e^Y] - 1) - 0.5 * σ²) dt + σ √(dt) Z) * (product of jumps)
where N(dt) ~ Poisson(λ dt) and Y_i ~ N(μ_J, σ_J).

Methods
-------
- sample_paths: Simulate asset paths.
- sample_paths_and_derivative: Simulate asset paths and compute ∂S/∂S0.

Usage
-----
    from sde.models.merton_jump_sde import MertonJumpSDE
    sde_sim = MertonJumpSDE(sigma, jump_intensity, muJ, sigmaJ)
    paths = sde_sim.sample_paths(T, n_sims, n_steps, r, q, S0)
    paths, dpaths_dS0 = sde_sim.sample_paths_and_derivative(...)
"""

import math
import numpy as np
from typing import Optional, Tuple


class MertonJumpSDE:
    """
    SDE simulator for the Merton jump-diffusion model.

    Parameters
    ----------
    sigma : float
        Diffusive volatility.
    jump_intensity : float
        Jump intensity (λ).
    muJ : float
        Mean jump size (in log-space).
    sigmaJ : float
        Volatility of jump sizes.
    random_state : Optional[int], optional
        Seed for random number generation (default None).
    """

    def __init__(
        self,
        sigma: float,
        jump_intensity: float,
        muJ: float,
        sigmaJ: float,
        random_state: Optional[int] = None,
    ) -> None:
        self.sigma = sigma
        self.jump_intensity = jump_intensity
        self.muJ = muJ
        self.sigmaJ = sigmaJ
        self._rng = np.random.default_rng(random_state)

    def sample_paths(
        self, T: float, n_sims: int, n_steps: int, r: float, q: float, S0: float
    ) -> np.ndarray:
        """
        Simulate asset paths under Merton jump-diffusion using a log-Euler scheme.

        Parameters
        ----------
        T : float
            Time to maturity in years.
        n_sims : int
            Number of simulation paths.
        n_steps : int
            Number of time steps.
        r : float
            Risk-free rate.
        q : float
            Dividend yield.
        S0 : float
            Initial asset price.

        Returns
        -------
        np.ndarray
            Array of simulated asset paths with shape (n_sims, n_steps+1).
        """
        dt = T / n_steps
        S = np.zeros((n_sims, n_steps + 1), dtype=float)
        S[:, 0] = S0

        # Precompute E[e^Y]
        EJ = math.exp(self.muJ + 0.5 * self.sigmaJ**2)
        drift = (r - q - self.jump_intensity * (EJ - 1) - 0.5 * self.sigma**2) * dt
        vol = self.sigma * math.sqrt(dt)

        for step in range(n_steps):
            Z = self._rng.normal(size=n_sims)
            log_s_k = np.log(S[:, step] + 1e-16)
            log_s_next = log_s_k + drift + vol * Z
            S_next = np.exp(log_s_next)
            # jumps
            N_jumps = self._rng.poisson(self.jump_intensity * dt, size=n_sims)
            for i in range(n_sims):
                if N_jumps[i] > 0:
                    Y = self._rng.normal(self.muJ, self.sigmaJ, size=N_jumps[i])
                    S_next[i] *= math.exp(Y.sum())
            S[:, step + 1] = S_next
        return S

    def sample_paths_and_derivative(
        self, T: float, n_sims: int, n_steps: int, r: float, q: float, S0: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate asset paths under Merton jump-diffusion and compute ∂S/∂S0.

        Parameters
        ----------
        T : float
            Time to maturity in years.
        n_sims : int
            Number of simulation paths.
        n_steps : int
            Number of time steps.
        r : float
            Risk-free rate.
        q : float
            Dividend yield.
        S0 : float
            Initial asset price.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (S, dSdS0) where:
            - S is shape (n_sims, n_steps+1) of simulated paths.
            - dSdS0 is shape (n_sims, n_steps+1) of derivatives ∂S(t)/∂S0.
        """
        dt = T / n_steps
        S = np.zeros((n_sims, n_steps + 1), dtype=float)
        dSdS0 = np.zeros((n_sims, n_steps + 1), dtype=float)

        S[:, 0] = S0
        dSdS0[:, 0] = 1.0

        EJ = math.exp(self.muJ + 0.5 * self.sigmaJ**2)
        drift = (r - q - self.jump_intensity * (EJ - 1) - 0.5 * self.sigma**2) * dt
        vol = self.sigma * math.sqrt(dt)

        for step in range(n_steps):
            Z = self._rng.normal(size=n_sims)
            log_s_k = np.log(S[:, step] + 1e-16)
            log_s_next = log_s_k + drift + vol * Z
            S_next = np.exp(log_s_next)
            ratio = S_next / (S[:, step] + 1e-16)

            # apply ratio to derivative
            dS_temp = dSdS0[:, step] * ratio

            # handle jumps
            N_jumps = self._rng.poisson(self.jump_intensity * dt, size=n_sims)
            for i in range(n_sims):
                if N_jumps[i] > 0:
                    Y = self._rng.normal(self.muJ, self.sigmaJ, size=N_jumps[i])
                    jump_factor = math.exp(Y.sum())
                    S_next[i] *= jump_factor
                    dS_temp[i] *= jump_factor

            S[:, step + 1] = S_next
            dSdS0[:, step + 1] = dS_temp

        return S, dSdS0

    def __repr__(self) -> str:
        """
        Return a string representation of the MertonJumpSDE simulator.

        Returns
        -------
        str
            Representation including sigma, jump_intensity, muJ, and sigmaJ.
        """
        return (
            f"{self.__class__.__name__}(sigma={self.sigma}, "
            f"jump_intensity={self.jump_intensity}, muJ={self.muJ}, "
            f"sigmaJ={self.sigmaJ})"
        )
