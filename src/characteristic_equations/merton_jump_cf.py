#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
merton_jump_cf.py
=================
Provides the characteristic function for the Merton jump-diffusion model.

The characteristic function for the log-price in Merton's jump-diffusion model is:
    φ(u) = exp{ i*u*(ln(S) + (r - q - λκ - 0.5σ²)*t)
              - 0.5σ²*t*u²
              + λ*t*(exp(i*u*μ_J - 0.5σ_J²*u²) - 1) }
where κ = exp(μ_J + 0.5σ_J²) - 1.

Usage:
    from characteristic_equations.merton_jump_cf import merton_jump_cf
    cf = merton_jump_cf(t, S, r, q, sigma, jump_intensity, muJ, sigmaJ)
    value = cf(u)
"""

import math
import numpy as np
from typing import Callable


def merton_jump_cf(
    t: float,
    spot: float,
    r: float,
    q: float,
    sigma: float,
    jump_intensity: float,
    muJ: float,
    sigmaJ: float,
) -> Callable[[complex], complex]:
    """
    Compute the characteristic function for the Merton jump-diffusion model.

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
        Diffusive volatility.
    jump_intensity : float
        Jump intensity (λ).
    muJ : float
        Mean jump size (in log space).
    sigmaJ : float
        Volatility of jump sizes.

    Returns
    -------
    Callable[[complex], complex]
        A function φ(u) computing the characteristic function value for a given complex number u.

    Examples
    --------
    >>> cf = merton_jump_cf(1.0, 100, 0.05, 0.02, 0.2, 0.1, -0.1, 0.3)
    >>> abs(cf(1.0)) > 0
    True
    """
    # Compute kappa = exp(μ_J + 0.5σ_J²) - 1
    kappa = math.exp(muJ + 0.5 * sigmaJ**2) - 1
    drift_adjustment = r - q - jump_intensity * kappa - 0.5 * sigma**2

    def phi(u: complex) -> complex:
        cont = 1j * u * (
            math.log(spot) + drift_adjustment * t
        ) - 0.5 * sigma**2 * t * (u**2)
        jump = (
            jump_intensity * t * (np.exp(1j * u * muJ - 0.5 * sigmaJ**2 * (u**2)) - 1)
        )
        return np.exp(cont + jump)

    return phi
