from typing import Optional
import math
from src.stochastic.sde.base_sde_extended import BaseModelExtended
import numpy as np


#######################################################
# 3) CIR
# RATE
#######################################################
class CIR(BaseModelExtended):
    """
    dr = a(b- r) dt + sigma sqrt(r) dW.
    We'll do a naive Euler that can go negative if step is large.
    partial(r_{k+1}, r_k)= ...
    We'll approximate partial step= 1 - a dt + partial( sqrt(r_k) Z ) ???

    This can be tricky to do exactly. We'll do a naive approach ignoring derivative of sqrt(r_k).
    For real usage, do partial-truncation or exact approach.
    """

    def __init__(self, r0, a, b, sigma, random_state=None):
        super().__init__(r=0.0, q=0.0, S0=r0, random_state=random_state)
        self.a = a
        self.b = b
        self.sigma = sigma

    def sample_paths(self, T, n_sims, n_steps):
        dt = T / n_steps
        r_arr = np.zeros((n_sims, n_steps + 1))
        r_arr[:, 0] = self.S0
        for step in range(n_steps):
            Z = self._rng.normal(size=n_sims)
            r_k = r_arr[:, step]
            drift = self.a * (self.b - r_k) * dt
            diff = self.sigma * np.sqrt(np.maximum(r_k, 0)) * math.sqrt(dt) * Z
            r_next = r_k + drift + diff
            # no positivity clamp => can go negative
            r_arr[:, step + 1] = r_next
        return r_arr

    def sample_paths_and_derivative(self, T, n_sims, n_steps):
        dt = T / n_steps
        r_arr = np.zeros((n_sims, n_steps + 1))
        drdr0 = np.zeros((n_sims, n_steps + 1))
        r_arr[:, 0] = self.S0
        drdr0[:, 0] = 1.0

        for step in range(n_steps):
            Z = self._rng.normal(size=n_sims)
            r_k = r_arr[:, step]
            dr_k = drdr0[:, step]

            drift = self.a * (self.b - r_k) * dt
            diff = self.sigma * np.sqrt(np.maximum(r_k, 0)) * math.sqrt(dt) * Z
            r_next = r_k + drift + diff

            # partial wrt r_k = 1 - a dt + partial( sqrt(r_k)*Z ) ???
            # naive approach ignoring derivative of sqrt(r_k) => 1 - a dt
            partial_step = 1.0 - self.a * dt
            dr_next = dr_k * partial_step

            r_arr[:, step + 1] = r_next
            drdr0[:, step + 1] = dr_next
        return r_arr, drdr0

    def __repr__(self):
        base = super().__repr__()
        return f"{base}, CIR(a={self.a}, b={self.b}, sigma={self.sigma})"
