"""
option_pricing.py
=================

Monte Carlo pricing of European and American options using the SDE models
from sde_models.py.

Features:
---------
1) price_european(...) - straightforward discount of payoff
2) price_american_lsm(...) - a least-squares MC approach
3) rational_bounds_check(...) - ensures no violation of trivial bounds
"""

import numpy as np
from typing import Callable
from src.stochastic.base_sde import BaseSDEModel


def rational_bounds_check(
    price: float, S0: float, K: float, r: float, q: float, T: float, is_call: bool
) -> float:
    """
    Enforce rational lower bound on the option price.
    For a call: price >= max(0, S0 e^{-qT} - K e^{-rT})
    For a put : price >= max(0, K e^{-rT} - S0 e^{-qT})
    """
    disc_factor_r = np.exp(-r * T)
    disc_factor_q = np.exp(-q * T)
    if is_call:
        lower_bound = max(0.0, S0 * disc_factor_q - K * disc_factor_r)
    else:
        lower_bound = max(0.0, K * disc_factor_r - S0 * disc_factor_q)

    return max(price, lower_bound)


def price_european(
    model: BaseSDEModel,
    payoff_fn: Callable[[np.ndarray], np.ndarray],
    T: float,
    n_sims: int,
    n_steps: int,
) -> float:
    """
    Simple MC for a European payoff: E[ e^{-rT} payoff(S_T) ].

    Parameters
    ----------
    model : BaseSDEModel
        The SDE model providing sample_paths(...).
    payoff_fn : callable
        payoff_fn(S_T) -> payoff value per path at maturity.
    T : float
        Time to maturity.
    n_sims : int
        # of Monte Carlo simulations.
    n_steps : int
        # of time steps for sample_paths.

    Returns
    -------
    float
        The MC estimate of the option price.
    """
    paths = model.sample_paths(T, n_sims, n_steps)
    ST = paths[:, -1]  # final column
    payoffs = payoff_fn(ST)
    disc_factor = np.exp(-model.r * T)
    price = disc_factor * payoffs.mean()
    return price


def price_american_lsm(
    model: BaseSDEModel,
    payoff_fn: Callable[[np.ndarray], np.ndarray],
    T: float,
    n_sims: int,
    n_steps: int,
) -> float:
    """
    Basic Longstaff–Schwartz approach (vectorized) for American option.
    Steps:
      1) sample full paths: shape (n_sims, n_steps+1)
      2) at each step from the back, decide whether to continue or to exercise,
         based on a regression of in-the-money paths.

    payoff_fn: payoff(S) -> immediate payoff.

    Returns
    -------
    float
        The LSM estimate of the option price.
    """
    dt = T / n_steps
    discount = np.exp(-model.r * dt)

    paths = model.sample_paths(T, n_sims, n_steps)
    # payoffs at each time step
    pay_matrix = payoff_fn(paths)  # but payoff_fn might only expect final S
    # we can adapt payoff_fn to accept arrays. Let's do a quick approach:
    # If payoff is something like a call: payoff = np.maximum(S-K,0).
    # We'll define payoff_fn s.t. it can handle the entire 2D array.

    # For LSM:
    # We'll store an array "cashflow" of shape (n_sims, n_steps+1),
    # which will hold the optimal cashflow at each step (in backwards induction).
    cashflow = np.zeros_like(paths)
    cashflow[:, -1] = pay_matrix[:, -1]  # final time is just exercise payoff

    # backward induction
    for step in range(n_steps - 1, 0, -1):
        in_money = pay_matrix[:, step] > 1e-14  # or some condition
        # Regress discount * CF_{step+1} on basis polynomials of S
        X = paths[in_money, step]
        Y = cashflow[in_money, step + 1] * discount
        if len(X) > 0:
            # polynomial fit e.g. [1, S, S^2]
            # simplest approach
            poly = np.column_stack([np.ones(len(X)), X, X**2])
            coeff, _, _, _ = np.linalg.lstsq(poly, Y, rcond=None)
            # predicted continuation
            continuation = coeff[0] + coeff[1] * X + coeff[2] * (X**2)
            # compare immediate exercise vs. continuation
            exercise_val = pay_matrix[in_money, step]
            # if exercise_val>continuation => exercise
            ex_indices = exercise_val > continuation
            # update those paths
            idx_all = np.where(in_money)[0]
            ex_mask = idx_all[ex_indices]
            cashflow[ex_mask, step] = exercise_val[ex_indices]
            # for others, keep step+1 CF discounted
            hold_mask = idx_all[~ex_indices]
            cashflow[hold_mask, step] = cashflow[hold_mask, step + 1] * discount
        else:
            # no paths in money => do nothing
            cashflow[:, step] = cashflow[:, step + 1] * discount

    # at step=0, discount forward
    price = cashflow[:, 1].mean() * discount  # we do discount 1 more step
    # or equivalently just do step=0 logic if in money
    # This is a rough, typical LSM approach.

    return price


# Example: If we define a call payoff
def call_payoff(S: np.ndarray, K=100.0):
    return np.maximum(S - K, 0.0)


def put_payoff(S: np.ndarray, K=100.0):
    return np.maximum(K - S, 0.0)
