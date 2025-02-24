#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
parity_implied_rate.py
======================
Provides a utility function for computing the implied risk-free rate from put-call
parity for European (and dividend-less American) options.

Functions
---------
implied_rate(call_price, put_price, S, K, T, q=0.0, eps=1e-6, max_iter=100)
    Numerically solves for the implied risk-free rate using the Brent method.
"""

import math
from typing import Union
from scipy.optimize import brentq


def implied_rate(
    call_price: float,
    put_price: float,
    S: float,
    K: float,
    T: float,
    q: Union[float, None] = 0.0,
    eps: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """
    Compute the implied risk-free rate from put-call parity.

    For a European option with continuous dividend yield q, put-call parity states:
        C - P = S * exp(-q * T) - K * exp(-r * T)
    The function solves for r such that:
        f(r) = S * exp(-q * T) - K * exp(-r * T) - (C - P) = 0

    Parameters
    ----------
    call_price : float
        Observed call price.
    put_price : float
        Observed put price.
    S : float
        Underlying spot price.
    K : float
        Strike price.
    T : float
        Time to maturity (in years).
    q : float or None, optional
        Continuous dividend yield (default is 0.0).
    eps : float, optional
        Convergence tolerance for the root-finder (default is 1e-6).
    max_iter : int, optional
        Maximum iterations for bracketing (default is 100).

    Returns
    -------
    float
        The implied risk-free rate.

    Raises
    ------
    ValueError
        If a valid bracket for r cannot be found within max_iter iterations.

    Examples
    --------
    >>> implied_rate(0.5287, 6.7143, 100, 110, 0.5, q=0.01)
    0.07999981808260372
    """
    if q is None:
        q = 0.0

    left_side = call_price - put_price
    discounted_S = S * math.exp(-q * T)

    def f(r: float) -> float:
        return discounted_S - K * math.exp(-r * T) - left_side

    # Set initial bracket for r.
    r_low, r_high = -1.0, 1.0
    f_low, f_high = f(r_low), f(r_high)
    iter_count = 0

    # Expand the bracket until a sign change is found.
    while f_low * f_high > 0:
        if abs(f_low) < abs(f_high):
            r_low -= 0.5
            f_low = f(r_low)
            if r_low < -100:
                raise ValueError(
                    "Cannot bracket negative rate further. Possibly no solution."
                )
        else:
            r_high += 0.5
            f_high = f(r_high)
            if r_high > 2.0:
                raise ValueError(
                    "Cannot bracket positive rate further. Possibly no solution."
                )
        iter_count += 1
        if iter_count > max_iter:
            raise ValueError(
                "Max iterations reached while bracketing for implied rate."
            )

    r_est = brentq(f, r_low, r_high, xtol=eps, maxiter=max_iter)
    return r_est
