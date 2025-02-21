#######################################################
# 1) Vasicek
# RATE
#######################################################
from typing import Optional
from src.stochastic.sde.base_sde_extended import BaseModelExtended
import numpy as np


class Vasicek(BaseModelExtended):
    """
    dr = a(b - r) dt + sigma dW. r can be negative.

    For derivative, partial(r_{k+1}, r_k)= (1 - a dt).
    Then chain for partial wrt r0.
    """

    def __init__(
        self,
        r0: float,
        a: float,
        b: float,
        sigma: float,
        random_state: Optional[int] = None,
    ):
        # We'll store r=..., q=0, S0=r0 in super so we don't break the base code
        super().__init__(r=0.0, q=0.0, S0=r0, random_state=random_state)
        self.a = a
        self.b = b
        self.sigma = sigma

    def sample_paths(self, T: float, n_sims: int, n_steps: int) -> np.ndarray:
        dt = T / n_steps
        r_arr = np.zeros((n_sims, n_steps + 1), dtype=float)
        r_arr[:, 0] = self.S0

        for step in range(n_steps):
            Z = self._rng.normal(size=n_sims)
            r_k = r_arr[:, step]
            drift = self.a * (self.b - r_k) * dt
            diff = self.sigma * np.sqrt(dt) * Z
            r_next = r_k + drift + diff
            r_arr[:, step + 1] = r_next
        return r_arr

    def sample_paths_and_derivative(self, T: float, n_sims: int, n_steps: int):
        dt = T / n_steps
        r_arr = np.zeros((n_sims, n_steps + 1), dtype=float)
        drdr0 = np.zeros((n_sims, n_steps + 1), dtype=float)

        r_arr[:, 0] = self.S0
        drdr0[:, 0] = 1.0

        for step in range(n_steps):
            Z = self._rng.normal(size=n_sims)
            r_k = r_arr[:, step]
            dr_k = drdr0[:, step]

            drift = self.a * (self.b - r_k) * dt
            diff = self.sigma * np.sqrt(dt) * Z
            r_next = r_k + drift + diff

            # partial(r_next, r_k)= 1 - a dt
            partial_step = 1.0 - self.a * dt
            dr_next = dr_k * partial_step

            r_arr[:, step + 1] = r_next
            drdr0[:, step + 1] = dr_next
        return r_arr, drdr0

    def __repr__(self):
        base = super().__repr__()
        return f"{base}, Vasicek(a={self.a}, b={self.b}, sigma={self.sigma})"
