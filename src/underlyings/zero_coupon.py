"""
rate_models.py

Demonstrates OOP Monte Carlo for interest-rate models that can go negative:

1) Vasicek
2) HullWhite (Extended Vasicek)
3) CIR (we allow negative if Euler steps cause it, or clamp if we prefer)
4) G2++ (two-factor Gaussian)

We inherit from BaseModelExtended (existing code) but do NOT modify it.

A sample usage function price_zcb(...) shows how to price a zero-coupon bond
via E[ exp(-integral r(t) dt ) ] in discrete form.
"""

#######################################################
# zero-coupon bond payoff
#######################################################
import numpy as np
from src.stochastic.sde.base_sde_extended import BaseModelExtended


# We assume your existing sde_models_extended has a base class "BaseModelExtended"
# that requires: sample_paths(...), sample_paths_and_derivative(...)
def price_zcb(model: BaseModelExtended, T: float, n_sims: int, n_steps: int) -> float:
    """
    Price a zero-coupon bond paying 1 at time T via:
      ZCB = E[exp(-∫ r(t) dt )].
    We'll do a discrete sum: discount_factor_i = exp(- dt * sum(r_i)).
    Return average discount_factor.

    model.sample_paths(...) => shape (n_sims, n_steps+1) of short rates r(t).

    T is total horizon, dt = T/n_steps
    """
    dt = T / n_steps
    r_paths = model.sample_paths(T, n_sims, n_steps)  # short rates
    # discrete approx integral r(t) dt => sum(r_k)*dt
    # each row => discount_factor = exp(- dt * sum(r_k from k=1..n_steps))
    # ignoring r_k at 0 if you want...
    # We'll do sum from 1..n_steps.
    sum_rates = np.sum(r_paths[:, 1:], axis=1)  # shape (n_sims,)
    discount_factor = np.exp(-dt * sum_rates)
    return discount_factor.mean()
