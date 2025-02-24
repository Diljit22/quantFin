#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
kou_cf.py
=========
Provides the characteristic function for the Kou (double-exponential) jump-diffusion model.

Kou's model extends Black-Scholes by adding a Poisson jump process where:
  - Each jump size Y ~ p * Exp( alpha1 ) on the upside
    + (1-p) * Exp( alpha2 ) on the downside (with sign).
In log-terms, upward jumps are log(1 + jump), downward are log(1 - jump?), or the distribution is
often parameterized so that the log-price jump is a mixture of exponentials.

Characteristic Function
-----------------------
Under the risk-neutral measure, the log-price X_t = ln(S_t) has CF:

.. math::
    \\phi_{Kou}(u) = \\exp\\Bigl( i u [\\ln(S_0) + (r - q - \\lambda \\kappa_J) t]
                                 - 0.5 \\sigma^2 u^2 t
                                 + \\lambda t [ G_{Kou}(u) ] \\Bigr),

where :math:`\\lambda` is jump intensity, and

.. math::
    \\kappa_J = E[e^{Y}] - 1, \\quad
    G_{Kou}(u) = E[e^{i u Y}] - 1,
    E[e^{i u Y}] = p \\frac{\\alpha_1}{\\alpha_1 - i u}
                 + (1 - p) \\frac{\\alpha_2}{\\alpha_2 + i u},

depending on sign convention. The final form is:

.. math::
    \\phi_{Kou}(u) = \\exp\\Bigl( i u [\\ln(S_0) + (r - q) t] - 0.5 \\sigma^2 t u^2
      + \\lambda t\\bigl( \\frac{p \\alpha_1}{\\alpha_1 - i u}
                         + \\frac{(1-p) \\alpha_2}{\\alpha_2 + i u} - 1 \\bigr ) \\Bigr)

if :math:`\\kappa_J = p \\frac{\\alpha_1}{\\alpha_1} + (1 - p) \\dots - 1`.

Usage
-----
    from characteristic_equations.kou_cf import kou_cf
    cf = kou_cf(t=1.0, spot=100, r=0.05, q=0.02,
                sigma=0.2, jump_intensity=0.1, p=0.3, alpha1=5.0, alpha2=5.0)
    val = cf(1.0 + 0j)
    print(val)
"""

import math
import cmath
from typing import Callable


def kou_cf(
    t: float,
    spot: float,
    r: float,
    q: float,
    sigma: float,
    jump_intensity: float,
    p_up: float,
    alpha1: float,
    alpha2: float,
) -> Callable[[complex], complex]:
    r"""
    Compute the characteristic function of the Kou double-exponential jump-diffusion model.

    Parameters
    ----------
    t : float
        Time to maturity in years.
    spot : float
        Current spot price (S0).
    r : float
        Risk-free rate.
    q : float
        Continuous dividend yield.
    sigma : float
        Diffusive volatility.
    jump_intensity : float
        Poisson jump intensity (lambda).
    p_up : float
        Probability of upward jump. Must be in [0,1].
    alpha1 : float
        Rate for the upward jump size distribution (exponential).
    alpha2 : float
        Rate for the downward jump distribution. Usually alpha2 > 0.

    Returns
    -------
    Callable[[complex], complex]
        A function phi(u) that computes the CF at a complex u.

    Notes
    -----
    The jump size Y is a mixture:
      - with prob p_up, Y ~ Exp(alpha1) for upward
      - with prob (1 - p_up), Y ~ -Exp(alpha2) for downward
    or equivalently log(1 + x) approach. Different references exist, but we assume
    a simpler log-jump approach here with mixture exponentials. The CF is:

    .. math::
        \phi_{Kou}(u) = \exp \Bigl( i u [\ln(S_0) + (r - q - \lambda \kappa_J) t ]
                                    - 0.5 \sigma^2 u^2 t
                                    + \lambda t [ M_Y(i u) - 1 ]\Bigr),

    where M_Y(i u) is the mgf of the jump distribution at i u. For double-exponential:

    .. math::
        M_Y(i u) = \frac{p_up \, \alpha_1}{\alpha_1 - i u}
                 + \frac{(1 - p_up)\, \alpha_2}{\alpha_2 + i u}.

    The drift correction \(\kappa_J\) ensures the process is martingale.

    Raises
    ------
    ValueError
        If sigma < 0, alpha1 < 0, alpha2 < 0, p_up not in [0,1].
    """
    if sigma < 0:
        raise ValueError("sigma must be non-negative.")
    if not 0.0 <= p_up <= 1.0:
        raise ValueError("p_up must be in [0,1].")
    if alpha1 <= 0 or alpha2 <= 0:
        raise ValueError("alpha1, alpha2 must be > 0.")
    if jump_intensity < 0:
        raise ValueError("jump_intensity must be >= 0.")

    i = 1j
    lnS = math.log(spot)

    # Expectation E[e^Y]:
    #   E[e^Y] = p_up * (alpha1/(alpha1-1)) if imaginary? Actually for i= -1. We do direct approach:
    # For double-expo, a standard formula is:
    #   E[e^Y] = p_up * alpha1/(alpha1-1) + (1 - p_up) * alpha2/(alpha2+1)  if it is "Y ~ mixture"
    # We'll define them carefully:

    # partial mgf of Y at 1 => E[e^Y].
    #   E[e^Y] = p_up * alpha1/(alpha1 - 1) + (1-p_up) * alpha2/(alpha2 + 1),
    # provided alpha1>1, alpha2>1 for integrability if we assume Y can be negative for the second part.
    # We'll not do heavy checks for domain; assume user knows model.

    # We'll define kappaJ = E[e^Y] - 1
    E_eY = p_up * (alpha1 / (alpha1 - 1.0)) + (1.0 - p_up) * (alpha2 / (alpha2 + 1.0))
    kappaJ = E_eY - 1.0

    def phi(u: complex) -> complex:
        # drift correction
        drift_term = i * u * (lnS + (r - q - jump_intensity * kappaJ) * t)
        diff_part = -0.5 * sigma * sigma * (u**2) * t

        # M_Y(i u):
        # M_Y(i u) = p_up * alpha1/(alpha1 - i u) + (1-p_up)* alpha2/(alpha2 + i u)
        # Then add -1, multiply by lambda t
        mgf_jump = p_up * (alpha1 / (alpha1 - i * u)) + (1.0 - p_up) * (
            alpha2 / (alpha2 + i * u)
        )
        jump_factor = jump_intensity * t * (mgf_jump - 1.0)

        return cmath.exp(drift_term + diff_part + jump_factor)

    return phi
