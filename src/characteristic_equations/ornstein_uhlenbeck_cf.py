#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ornstein_uhlenbeck_cf.py
========================
Characteristic function for the Ornstein–Uhlenbeck (OU) process:

    dX_t = kappa (theta - X_t) dt + sigma dW_t.

We can solve X_t analytically:

    X_t = X_0 e^{-kappa t} + theta (1 - e^{-kappa t})
          + sigma e^{-kappa t} \int_0^t e^{kappa s} dW_s.

Hence X_t ~ Normal with mean and variance known. The characteristic function
of X_t is straightforward:

    E[e^{i u X_t}] = exp( i u m(t) - 0.5 u^2 v(t) ),

where
  m(t) = X_0 e^{-kappa t} + theta (1 - e^{-kappa t}),
  v(t) = (sigma^2 / (2 kappa)) (1 - e^{-2 kappa t}).

References
----------
- Uhlenbeck, G. E., & Ornstein, L. S. (1930). "On the theory of the Brownian motion."

Usage
-----
    from characteristic_equations.ornstein_uhlenbeck_cf import ou_cf
    cf = ou_cf(t=1.0, X0=1.0, kappa=1.5, theta=2.0, sigma=0.3)
    val = cf(1.0+0j)
    print(val)
"""

import math
import cmath
from typing import Callable

def ou_cf(
    t: float,
    X0: float,
    kappa: float,
    theta: float,
    sigma: float
) -> Callable[[complex], complex]:
    r"""
    Characteristic function for the OU process X_t:

    X_t ~ Normal( m(t), v(t) ), where
      m(t) = X0 e^{-kappa t} + theta (1 - e^{-kappa t}),
      v(t) = \frac{\sigma^2}{2 kappa} (1 - e^{-2 kappa t}).

    Then,

    .. math::
       \phi_{X_t}(u) = \exp\left( i u \, m(t) \;-\; \tfrac12 \, u^2 \, v(t) \right).

    Parameters
    ----------
    t : float
        Time in years.
    X0 : float
        Initial level X(0).
    kappa : float
        Mean reversion speed, > 0.
    theta : float
        Long-run mean.
    sigma : float
        Vol parameter, > 0.

    Returns
    -------
    Callable[[complex], complex]
        A function phi(u).

    Raises
    ------
    ValueError
        If kappa<=0 or sigma<=0.

    Example
    -------
    >>> cf = ou_cf(t=1.0, X0=1.0, kappa=1.5, theta=2.0, sigma=0.3)
    >>> val = cf(1.0+0j)
    >>> abs(val) > 0
    True
    """

    if kappa<=0.0 or sigma<=0.0:
        raise ValueError("kappa>0, sigma>0 required for Ornstein-Uhlenbeck.")
    m = X0*math.exp(-kappa*t) + theta*(1.0 - math.exp(-kappa*t))
    v = (sigma*sigma/(2.0*kappa))*(1.0 - math.exp(-2.0*kappa*t))

    def phi(u: complex) -> complex:
        i = 1j
        return cmath.exp(i*u*m - 0.5*(u**2)*v)

    return phi
