from typing import Optional
import math
from src.stochastic.sde.base_sde_extended import BaseModelExtended
import numpy as np


#######################################################
# 4) G2++ (Two-Factor Gaussian)
# RATE
#######################################################
class G2pp(BaseModelExtended):
    """
    G2++ => x(t), y(t) each OU, r(t)= phi(t) + x(t)+ y(t).
    We'll store only x,y. partial wrt x0,y0 => if we assume x0,y0 depend on r0?
    For simplicity, let's treat (x0,y0)=0 and r0= x0+y0 => we skip details.
    """

    def __init__(
        self,
        r0: float,
        a: float,
        b: float,
        sigma: float,
        eta: float,
        rho: float,
        phi: float = 0.0,
        random_state=None,
    ):
        """
        r(t)= phi + x(t)+ y(t).
        dx= -a x dt + sigma dW1
        dy= -b y dt + eta   dW2
        corr(dW1, dW2)= rho
        r0 => x0=  r0, y0=0 or we do half? We'll store x0= r0, y0= 0 for example.
        """
        super().__init__(r=0.0, q=0.0, S0=r0, random_state=random_state)
        self.a = a
        self.b = b
        self.sigma = sigma
        self.eta = eta
        self.rho = rho
        self.phi = phi

    def sample_paths(self, T, n_sims, n_steps):
        dt = T / n_steps
        x_arr = np.full((n_sims,), self.S0, dtype=float)  # we treat x0= r0
        y_arr = np.zeros((n_sims,), dtype=float)

        out = np.zeros((n_sims, n_steps + 1), dtype=float)
        out[:, 0] = self.phi + x_arr + y_arr

        sqrt_dt = math.sqrt(dt)
        for step in range(n_steps):
            Z1 = self._rng.normal(size=n_sims)
            Z2 = self._rng.normal(size=n_sims)
            dW1 = Z1
            dW2 = self.rho * Z1 + math.sqrt(1 - self.rho**2) * Z2

            x_next = x_arr + (-self.a * x_arr) * dt + self.sigma * sqrt_dt * dW1
            y_next = y_arr + (-self.b * y_arr) * dt + self.eta * sqrt_dt * dW2
            out[:, step + 1] = self.phi + x_next + y_next
            x_arr = x_next
            y_arr = y_next
        return out

    def sample_paths_and_derivative(self, T, n_sims, n_steps):
        """
        partial wrt. r0 => partial(x0, r0)= 1, partial(y0, r0)=0
        partial(x_{k+1}, x_k)= 1- a dt
        partial(y_{k+1}, y_k)= 1- b dt
        ignoring the correlation in partial wrt. x0 => chain rule is straightforward.
        We'll store x,y in a 2D array or keep them separate. We'll store only r(t) in out, ignoring partial formula details for demonstration.
        """
        dt = T / n_steps
        x_arr = np.full(n_sims, self.S0)
        dxdr0 = np.ones(n_sims)
        y_arr = np.zeros(n_sims)
        dydr0 = np.zeros(n_sims)

        out = np.zeros((n_sims, n_steps + 1))
        dout = np.zeros((n_sims, n_steps + 1))
        out[:, 0] = self.phi + x_arr + y_arr
        dout[:, 0] = 1.0  # partial( r(0), r0)= 1

        sqrt_dt = math.sqrt(dt)

        for step in range(n_steps):
            Z1 = self._rng.normal(size=n_sims)
            Z2 = self._rng.normal(size=n_sims)
            dW1 = Z1
            dW2 = self.rho * Z1 + math.sqrt(1 - self.rho**2) * Z2

            # x next
            x_next = x_arr + (-self.a * x_arr) * dt + self.sigma * sqrt_dt * dW1
            dx_next = dxdr0 + (-self.a) * dt * dxdr0

            # y next
            y_next = y_arr + (-self.b * y_arr) * dt + self.eta * sqrt_dt * dW2
            dy_next = dydr0 + (-self.b) * dt * dydr0

            r_next = self.phi + x_next + y_next
            dr_next = (
                dx_next + dy_next
            )  # partial(r, r0)= partial(x, r0)+ partial(y, r0)

            out[:, step + 1] = r_next
            dout[:, step + 1] = dr_next

            x_arr = x_next
            dxdr0 = dx_next
            y_arr = y_next
            dydr0 = dy_next

        return out, dout

    def __repr__(self):
        base = super().__repr__()
        return (
            f"{base}, G2++(a={self.a}, b={self.b}, sigma={self.sigma}, "
            f"eta={self.eta}, rho={self.rho}, phi={self.phi})"
        )
