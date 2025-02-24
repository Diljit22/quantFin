#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cev_sde.py
==========
Implements SDE simulation for the CEV (Constant Elasticity of Variance) model.

Under the CEV model, the volatility term is S_t^beta, for some beta in (0,1], typically.
Thus the dynamics are:
    dS_t = (r - q) * S_t dt + sigma * S_t^beta * dW_t.

We implement an Euler-Maruyama approach:
    S_{k+1} = S_k + (r - q)*S_k*dt + sigma * (S_k^beta) * sqrt(dt)*Z.

Methods
-------
- sample_paths
- sample_paths_and_derivative
"""

import math
import numpy as np
from typing import Optional, Tuple


class CEVSDE:
    """
    SDE simulator for the Constant Elasticity of Variance model.

    Parameters
    ----------
    sigma : float
        Base volatility parameter.
    beta : float
        The elasticity exponent. Often 0 < beta <= 1.
    random_state : int, optional
        Seed for the random number generator.
    """

    def __init__(
        self, sigma: float, beta: float, random_state: Optional[int] = None
    ) -> None:
        self.sigma = sigma
        self.beta = beta
        self._rng = np.random.default_rng(random_state)

    def sample_paths(
        self, T: float, n_sims: int, n_steps: int, r: float, q: float, S0: float
    ) -> np.ndarray:
        """
        Simulate asset paths under CEV using Euler-Maruyama.

        dS = (r - q)*S dt + sigma * (S^beta) dW

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
            Shape (n_sims, n_steps+1) of simulated paths.
        """
        dt = T / n_steps
        S = np.zeros((n_sims, n_steps + 1), dtype=float)
        S[:, 0] = S0

        for step in range(n_steps):
            Z = self._rng.normal(size=n_sims)
            S_k = S[:, step]
            drift = (r - q) * S_k * dt
            vol = self.sigma * (S_k**self.beta) * math.sqrt(dt)
            S[:, step + 1] = S_k + drift + vol * Z
            # Enforce positivity if needed (CEV can go to 0 if beta<1):
            S[:, step + 1] = np.maximum(S[:, step + 1], 1e-16)

        return S

    def sample_paths_and_derivative(
        self, T: float, n_sims: int, n_steps: int, r: float, q: float, S0: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate asset paths under CEV and compute ∂S/∂S0 via Euler-Maruyama.

        Parameters
        ----------
        T : float
            Time to maturity.
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
            (S, dSdS0) shapes (n_sims, n_steps+1).
        """
        dt = T / n_steps
        S = np.zeros((n_sims, n_steps + 1), dtype=float)
        dSdS0 = np.zeros((n_sims, n_steps + 1), dtype=float)

        S[:, 0] = S0
        dSdS0[:, 0] = 1.0

        for step in range(n_steps):
            Z = self._rng.normal(size=n_sims)
            S_k = S[:, step]
            dSdS0_k = dSdS0[:, step]

            drift = (r - q) * S_k * dt
            vol = self.sigma * (S_k**self.beta) * math.sqrt(dt)

            # S_{k+1} = S_k + drift + vol * Z
            # partial(S_{k+1}, S_k) = 1 + (r - q)*dt + sigma*beta*S_k^(beta-1)*sqrt(dt)*Z
            # (when S_k>0). We'll do it elementwise to handle S_k^(beta-1).
            partial = (
                1.0
                + (r - q) * dt
                + self.sigma
                * self.beta
                * (S_k ** (self.beta - 1.0))
                * math.sqrt(dt)
                * Z
            )

            S_next = S_k + drift + vol * Z
            S_next = np.maximum(S_next, 1e-16)

            dS0_next = dSdS0_k * partial
            # If S_next is forcibly floored to 1e-16,
            # partial might be questionable if the path hits zero.

            S[:, step + 1] = S_next
            dSdS0[:, step + 1] = dS0_next

        return S, dSdS0

    def __repr__(self) -> str:
        """
        Return a string representation of the CEVSDE simulator.

        Returns
        -------
        str
            Representation including sigma and beta.
        """
        return f"{self.__class__.__name__}(sigma={self.sigma}, beta={self.beta})"
