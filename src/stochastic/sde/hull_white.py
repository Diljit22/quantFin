#######################################################
# 2) Hull–White (Extended Vasicek)
# RATE
#######################################################
from typing import Optional
import math
from src.stochastic.sde.base_sde_extended import BaseModelExtended
import numpy as np


class HullWhite(BaseModelExtended):
    """
    dr = [theta(t) - a r] dt + sigma dW. For simplicity, we treat theta(t) as constant.
    partial step => 1 - a dt
    same logic as vasicek but offset in drift.
    """

    def __init__(
        self,
        r0: float,
        a: float,
        theta: float,
        sigma: float,
        random_state: Optional[int] = None,
    ):
        super().__init__(r=0.0, q=0.0, S0=r0, random_state=random_state)
        self.a = a
        self.theta = theta
        self.sigma = sigma

    def sample_paths(self, T, n_sims, n_steps):
        dt = T / n_steps
        r_arr = np.zeros((n_sims, n_steps + 1))
        r_arr[:, 0] = self.S0

        for step in range(n_steps):
            Z = self._rng.normal(size=n_sims)
            r_k = r_arr[:, step]
            drift = (self.theta - self.a * r_k) * dt
            diff = self.sigma * math.sqrt(dt) * Z
            r_next = r_k + drift + diff
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

            drift = (self.theta - self.a * r_k) * dt
            diff = self.sigma * math.sqrt(dt) * Z
            r_next = r_k + drift + diff

            partial_step = 1.0 - self.a * dt
            dr_next = dr_k * partial_step

            r_arr[:, step + 1] = r_next
            drdr0[:, step + 1] = dr_next
        return r_arr, drdr0

    def __repr__(self):
        base = super().__repr__()
        return f"{base}, HullWhite(a={self.a}, theta={self.theta}, sigma={self.sigma})"
