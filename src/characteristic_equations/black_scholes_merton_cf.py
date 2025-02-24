#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
black_scholes_merton_cf.py
==========================
Provides the characteristic function for the Black‐Scholes‐Merton model.

The characteristic function of the log‐price under BSM is given by:
    phi(u) = exp(i*u*(ln(spot) + (r - q - 0.5*sigma^2)*t) - 0.5*sigma^2*t*u^2)

Usage:
    from characteristic_equations.black_scholes_merton_cf import black_scholes_merton_cf
    cf = black_scholes_merton_cf(t, spot, r, q, sigma)
    value = cf(u)
"""

import math
import numpy as np
from typing import Callable


def black_scholes_merton_cf(
    t: float, spot: float, r: float, q: float, sigma: float
) -> Callable[[complex], complex]:
    """
    Compute the characteristic function for the Black‐Scholes‐Merton model.

    Parameters
    ----------
    t : float
        Time to maturity in years.
    spot : float
        Current spot price.
    r : float
        Risk-free interest rate.
    q : float
        Dividend yield.
    sigma : float
        Volatility of the underlying asset.

    Returns
    -------
    Callable[[complex], complex]
        A function phi(u) that computes the characteristic function value at u.

    Examples
    --------
    >>> cf = black_scholes_merton_cf(1.0, 100.0, 0.05, 0.02, 0.2)
    >>> abs(cf(1.0)) > 0
    True
    """
    half_var = 0.5 * sigma * sigma
    drift = math.log(spot) + (r - q - half_var) * t

    def phi(u: complex) -> complex:
        return np.exp((1j * u * drift) - (half_var * t * (u**2)))

    return phi
