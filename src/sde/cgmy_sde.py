#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cgmy_sde.py
===========
Approximate simulation for the CGMY process. The CGMY has infinite activity,
so an exact SDE approach is not as straightforward. We do a
truncated approach for large jumps only, ignoring or approximating small jumps.

We do:
  - For each dt, sample N_l ~ Poisson(lambda_l dt) for large jumps, add them individually.
  - Approximate small jumps with a normal or stable approach. (placeholder)

This is strictly an approximate scheme.

Methods
-------
- sample_paths
- sample_paths_and_derivative
"""

import math
import numpy as np
from typing import Optional, Tuple

class CGMYSDE:
    """
    Approximate SDE simulator for CGMY. Large jump approach + small-jump approximation.

    Parameters
    ----------
    C : float
    G : float
    M : float
    Y : float
    random_state : int, optional
        RNG seed
    """

    def __init__(
        self,
        C: float,
        G: float,
        M: float,
        Y: float,
        random_state: Optional[int] = None
    ) -> None:
        self.C = C
        self.G = G
        self.M = M
        self.Y = Y
        self._rng = np.random.default_rng(random_state)

    def sample_paths(
        self,
        T: float,
        n_sims: int,
        n_steps: int,
        r: float,
        q: float,
        S0: float
    ) -> np.ndarray:
        """
        Approximate simulation of CGMY. We'll do a naive approach:
          - Let dt = T/n_steps
          - For each step, we sample some "big jump" count from Poisson. For each big jump, draw size from e.g. a double-sided power-law or heuristic?
          - We also do a small-jump normal approximation with zero mean? (massive simplification).

        The result is only a placeholder. Real production code might do
        series expansions or advanced approaches.

        Parameters
        ----------
        T, n_sims, n_steps, r, q, S0
            standard
        Returns
        -------
        np.ndarray
            shape (n_sims, n_steps+1)
        """
        dt = T/n_steps
        S = np.zeros((n_sims, n_steps+1), dtype=float)
        S[:, 0] = S0

        # We won't do an explicit (r-q) drift here, relying on CF correction for risk-neutral measure.
        # If you wanted direct e^{(r-q)*dt}, you can add it to the log step.

        # big jump intensity (toy):
        # There's a known indefinite integral for the CGMY measure, but let's pick
        # a "large jump rate" lamL = some fraction of dt. We'll define an arbitrary scale.
        lam_big = 0.2 * self.C  # TOTALLY arbitrary placeholder

        # small jump normal stdev "approx" can also be guessed
        sigma_small = 0.1 * (self.C)  # again, purely guess.

        for step in range(n_steps):
            # big jump events
            N_big = self._rng.poisson(lam_big*dt, size=n_sims)
            # For each path, if N>0, we do random sign and magnitude from e.g. "Pareto or alpha stable"?
            # We'll do a simplistic approach: exponential tail. TOTALLY approximate.
            jump_factor = np.ones(n_sims, dtype=float)
            for i in range(n_sims):
                nJ = N_big[i]
                if nJ > 0:
                    total_jump = 0.0
                    for _ in range(nJ):
                        sign = 1.0 if self._rng.random()<0.5 else -1.0
                        # magnitude from e.g. Exp( rate= (G+M)/2 )?
                        # The real CGMY approach is more subtle. We'll pick:
                        magnitude = self._rng.exponential(1.0/((self.G + self.M)/2.0))
                        total_jump += sign*magnitude
                    jump_factor[i] = math.exp(total_jump)

            # small jump normal approx
            d_small = sigma_small*math.sqrt(dt)*self._rng.normal(0,1, size=n_sims)

            # net log increment
            incr = np.exp(d_small)*jump_factor
            S[:, step+1] = S[:, step]*incr

        return S

    def sample_paths_and_derivative(
        self,
        T: float,
        n_sims: int,
        n_steps: int,
        r: float,
        q: float,
        S0: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Same as sample_paths, but track partial(S/∂S0) via ratio approach.

        Returns
        -------
        (S, dSdS0)
        """
        dt = T/n_steps
        S = np.zeros((n_sims, n_steps+1), dtype=float)
        dSdS0 = np.zeros((n_sims, n_steps+1), dtype=float)
        S[:, 0] = S0
        dSdS0[:, 0] = 1.0

        lam_big = 0.2*self.C
        sigma_small = 0.1*self.C

        for step in range(n_steps):
            N_big = self._rng.poisson(lam_big*dt, size=n_sims)
            jump_factor = np.ones(n_sims, dtype=float)
            for i in range(n_sims):
                nJ = N_big[i]
                if nJ>0:
                    total_jump = 0.0
                    for _ in range(nJ):
                        sign = 1.0 if self._rng.random()<0.5 else -1.0
                        magnitude = self._rng.exponential(1.0/((self.G + self.M)/2.0))
                        total_jump += sign*magnitude
                    jump_factor[i] = math.exp(total_jump)

            d_small = sigma_small*math.sqrt(dt)*self._rng.normal(0,1, size=n_sims)
            incr = np.exp(d_small)*jump_factor
            S_next = S[:, step]*incr
            ratio = S_next/(S[:, step]+1e-16)
            dS0_next = dSdS0[:, step]*ratio

            S[:, step+1] = S_next
            dSdS0[:, step+1] = dS0_next

        return S, dSdS0

    def __repr__(self) -> str:
        """
        Return a representation for debugging.

        Returns
        -------
        str
        """
        return (f"{self.__class__.__name__}(C={self.C}, G={self.G}, M={self.M}, Y={self.Y})")
