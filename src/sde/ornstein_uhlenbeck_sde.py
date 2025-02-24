#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ornstein_uhlenbeck_sde.py
=========================
Implements an Euler–Maruyama or exact step for the Ornstein–Uhlenbeck process:

   dX_t = kappa (theta - X_t) dt + sigma dW_t.

We can do a direct exact step:
   X_{n+1} = X_{n} e^{-kappa dt} + theta (1 - e^{-kappa dt}) + sigma \sqrt{\frac{1 - e^{-2 kappa dt}}{2 kappa}} Z.

Methods
-------
- sample_paths
- sample_paths_and_derivative
"""

import math
import numpy as np
from typing import Optional, Tuple

class OrnsteinUhlenbeckSDE:
    """
    SDE simulator for Ornstein–Uhlenbeck:

      dX_t = kappa (theta - X_t) dt + sigma dW_t.

    We implement an exact stepping scheme:
      X_{n+1} = X_{n} e^{-kappa dt} + theta (1 - e^{-kappa dt})
                + sigma * sqrt( (1 - e^{-2 kappa dt}) / (2 kappa ) ) * Z_n.

    Parameters
    ----------
    kappa : float
        Mean reversion speed.
    theta : float
        Long-run mean.
    sigma : float
        Vol param.
    random_state : int, optional
        RNG seed.
    """

    def __init__(
        self,
        kappa: float,
        theta: float,
        sigma: float,
        random_state: Optional[int] = None
    ) -> None:
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self._rng = np.random.default_rng(random_state)

    def sample_paths(
        self,
        T: float,
        n_sims: int,
        n_steps: int,
        X0: float
    ) -> np.ndarray:
        """
        Simulate OU paths using exact step.

        Parameters
        ----------
        T : float
        n_sims : int
        n_steps : int
        X0 : float
            Initial value.

        Returns
        -------
        np.ndarray
            shape (n_sims, n_steps+1)
        """
        dt = T/n_steps
        X = np.zeros((n_sims, n_steps+1), dtype=float)
        X[:, 0] = X0

        e_kdt = math.exp(-self.kappa*dt)
        var_factor = (1.0 - math.exp(-2.0*self.kappa*dt))/(2.0*self.kappa)

        for step in range(n_steps):
            Z = self._rng.normal(0,1, size=n_sims)
            X_k = X[:, step]
            X_next = (X_k* e_kdt
                      + self.theta*(1.0 - e_kdt)
                      + self.sigma*math.sqrt(var_factor)*Z)
            X[:, step+1] = X_next

        return X

    def sample_paths_and_derivative(
        self,
        T: float,
        n_sims: int,
        n_steps: int,
        X0: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate OU paths and partial derivative wrt X0.

        The ratio approach is not directly applicable because the increment is not multiplicative.
        We'll do a direct partial derivative from the exact solution:
          X_{n+1} = e^{-kappa dt} X_n + ...
        => partial(X_{n+1} wrt X_n) = e^{-kappa dt}.
        => partial(X_{n+1} wrt X0) = e^{-kappa dt} partial(X_n wrt X0}.

        We can implement that recursively.

        Returns
        -------
        (X, dXdX0) shape (n_sims, n_steps+1).
        """
        dt = T/n_steps
        X = np.zeros((n_sims, n_steps+1), dtype=float)
        dXdX0 = np.zeros((n_sims, n_steps+1), dtype=float)

        X[:, 0] = X0
        dXdX0[:, 0] = 1.0

        e_kdt = math.exp(-self.kappa*dt)
        var_factor = (1.0 - math.exp(-2.0*self.kappa*dt))/(2.0*self.kappa)

        for step in range(n_steps):
            Z = self._rng.normal(0,1, size=n_sims)
            X_k = X[:, step]
            dXk = dXdX0[:, step]

            X_next = e_kdt*X_k + self.theta*(1.0 - e_kdt) + self.sigma*math.sqrt(var_factor)*Z
            dX_next = e_kdt*dXk  # partial wrt X0 is e^{-kappa dt} times partial for previous

            X[:, step+1] = X_next
            dXdX0[:, step+1] = dX_next

        return X, dXdX0

    def __repr__(self) -> str:
        """
        Debug repr
        """
        return (f"{self.__class__.__name__}(kappa={self.kappa}, theta={self.theta}, "
                f"sigma={self.sigma})")
