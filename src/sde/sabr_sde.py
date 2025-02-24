#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sabr_sde.py
===========
Simulates the SABR model:

    dF_t = alpha_t F_t^beta dW_1,
    d(alpha_t) = nu alpha_t dW_2,

with correlation corr(W_1, W_2)=rho. Usually we might add a drift for F_t
like (r - q) etc., but standard SABR for e.g. interest rate or FX might
store it differently. We'll keep it general for an asset price S.

If 'beta=1', it's effectively a lognormal approach. If 'beta=0', it's normal.

We handle partial derivative wrt S0 using ratio approach for F_t,
though alpha_t is separate.

Methods
-------
- sample_paths
- sample_paths_and_derivative
"""

import math
import numpy as np
from typing import Optional, Tuple


class SABRSDE:
    """
    SDE simulator for the standard SABR model.

    Parameters
    ----------
    alpha0 : float
        Initial volatility level (alpha(0)).
    beta : float
        Elasticity exponent in [0,1].
    rho : float
        Correlation in [-1,1].
    nu : float
        Vol of volatility.
    random_state : int, optional
        RNG seed.
    """

    def __init__(
        self,
        alpha0: float,
        beta: float,
        rho: float,
        nu: float,
        random_state: Optional[int] = None,
    ) -> None:
        self.alpha0 = alpha0
        self.beta = beta
        self.rho = rho
        self.nu = nu
        self._rng = np.random.default_rng(random_state)

    def sample_paths(
        self, T: float, n_sims: int, n_steps: int, F0: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate (F_t, alpha_t) under the SABR model (no drift on F).

        dF_t = alpha_t F_t^beta dW_1
        d(alpha_t) = nu * alpha_t dW_2
        corr(W_1, W_2)=rho.

        Euler approach.

        Parameters
        ----------
        T : float
            Time in years.
        n_sims : int
            Number of paths.
        n_steps : int
            Steps per path.
        F0 : float
            Initial forward or price.

        Returns
        -------
        (F, alpha)
            F shape (n_sims, n_steps+1),
            alpha shape (n_sims, n_steps+1).
        """
        dt = T / n_steps
        F = np.zeros((n_sims, n_steps + 1))
        alpha = np.zeros((n_sims, n_steps + 1))

        F[:, 0] = F0
        alpha[:, 0] = self.alpha0

        for step in range(n_steps):
            Z1 = self._rng.normal(0, 1, size=n_sims)
            Z2 = self._rng.normal(0, 1, size=n_sims)
            W1 = Z1
            W2 = self.rho * Z1 + math.sqrt(1.0 - self.rho**2) * Z2

            alpha_k = alpha[:, step]
            F_k = F[:, step]
            alpha_next = alpha_k + self.nu * alpha_k * math.sqrt(dt) * W2
            # Euler => alpha_next = alpha_k * exp(nu sqrt(dt) W2) is a log approach if we prefer
            # but let's do the linear approach. We keep alpha>0 => floor:
            alpha_next = np.maximum(alpha_next, 1e-16)

            # F update
            if abs(self.beta - 1.0) < 1e-14:
                # log approach for F if beta=1 => dF=alpha_k * F dt => do log-euler
                # but we'll do standard euler:
                dF = alpha_k * (F_k**self.beta) * math.sqrt(dt) * W1
                F_next = F_k + dF
            else:
                # general euler for F
                dF = alpha_k * (F_k**self.beta) * math.sqrt(dt) * W1
                F_next = F_k + dF

            F_next = np.maximum(F_next, 1e-16)

            alpha[:, step + 1] = alpha_next
            F[:, step + 1] = F_next

        return F, alpha

    def sample_paths_and_derivative(
        self, T: float, n_sims: int, n_steps: int, F0: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate (F, alpha) under SABR and compute partial(F/∂F0).

        Parameters
        ----------
        T : float
            Maturity.
        n_sims : int
            # of paths.
        n_steps : int
            steps.
        F0 : float
            initial forward.

        Returns
        -------
        (F, alpha, dFdF0)
            shapes (n_sims, n_steps+1).
        """
        dt = T / n_steps
        F = np.zeros((n_sims, n_steps + 1))
        alpha = np.zeros((n_sims, n_steps + 1))
        dFdF0 = np.zeros((n_sims, n_steps + 1))

        F[:, 0] = F0
        alpha[:, 0] = self.alpha0
        dFdF0[:, 0] = 1.0

        for step in range(n_steps):
            Z1 = self._rng.normal(0, 1, size=n_sims)
            Z2 = self._rng.normal(0, 1, size=n_sims)
            W1 = Z1
            W2 = self.rho * Z1 + math.sqrt(1.0 - self.rho**2) * Z2

            alpha_k = alpha[:, step]
            F_k = F[:, step]
            dFdF0_k = dFdF0[:, step]

            # alpha update (Euler)
            alpha_next = alpha_k + self.nu * alpha_k * math.sqrt(dt) * W2
            alpha_next = np.maximum(alpha_next, 1e-16)

            # F update
            dF = alpha_k * (F_k**self.beta) * math.sqrt(dt) * W1
            F_next = F_k + dF

            ratio = (F_next + 1e-16) / (F_k + 1e-16)
            dFdF0_next = dFdF0_k * ratio

            F_next = np.maximum(F_next, 1e-16)

            alpha[:, step + 1] = alpha_next
            F[:, step + 1] = F_next
            dFdF0[:, step + 1] = dFdF0_next

        return F, alpha, dFdF0

    def __repr__(self) -> str:
        """
        String representation for debugging.

        Returns
        -------
        str
        """
        return (
            f"{self.__class__.__name__}(alpha0={self.alpha0}, beta={self.beta}, "
            f"rho={self.rho}, nu={self.nu})"
        )
