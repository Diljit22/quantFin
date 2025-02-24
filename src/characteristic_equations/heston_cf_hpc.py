#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
heston_cf_hpc.py
============
Provides a 'Little Trap' HPC-style implementation of the characteristic function
for the Heston model.

The Heston model assumes that the underlying asset S_t follows:
    dS_t = S_t (r - q) dt + S_t sqrt(v_t) dW_1,
    dv_t = kappa (theta - v_t) dt + sigma sqrt(v_t) dW_2,
with correlation corr(dW_1, dW_2) = rho, and v_0 as the initial variance.

Characteristic Function
-----------------------
We define φ(u) = E[exp(i u ln(S_t))], under the appropriate risk-neutral measure.
The Heston CF typically uses piecewise-defined 'trick' or 'trap' versions for numerical
stability. This is a HPC (High-Precision/“Little Trap”) style variant, referencing
common formula expansions from:
  - Steven L. Heston (1993), "A Closed-Form Solution for Options with Stochastic
    Volatility with Applications to Bond and Currency Options."
  - Albrecher and co., "The Little Heston Trap" approach, or
    Lord and Kahl's method for stable evaluation.

Usage
-----
    from characteristic_equations.heston_cf import heston_cf_hpc
    phi = heston_cf_hpc(
        t=1.0,
        spot=100.0,
        r=0.05,
        q=0.02,
        v0=0.04,
        kappa=1.5,
        theta=0.04,
        sigma=0.3,
        rho=-0.7,
        trap=1
    )
    value = phi(1.0 + 0j)  # Evaluate CF at u=1

References
----------
- Heston, S. (1993). "A Closed-Form Solution for Options with Stochastic Volatility
  with Applications to Bond and Currency Options," Review of Financial Studies, 6(2).
- Lord, R., & Kahl, C. (2010). "Complex logarithms in Heston-like models and
  their application to exponential Lévy models." 
- Albrecher, P., et al. "The Little Heston Trap."

Notes
-----
'trap' can be 1 (Little Trap) or 0 (Original Heston). Setting trap=1 typically
improves numerical stability.

"""

import cmath
import math
from typing import Callable


def heston_cf_hpc(
    t: float,
    spot: float,
    r: float,
    q: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    trap: int = 1,
) -> Callable[[complex], complex]:
    r"""
    Compute the HPC ("Little Trap") style characteristic function for the Heston model.

    The Heston model for log-price :math:`X_t = \ln(S_t)` has the characteristic function
    usually written as:

    .. math::
        \phi(u) = \exp\bigl(i\,u\,(\ln(S_0) + (r - q)\,t)\bigr)\,\exp\bigl(A(t,u) + B(t,u)\,v_0\bigr),

    where :math:`v_0` is the initial variance. The exact definitions of :math:`A(t,u)` and
    :math:`B(t,u)` differ in "trap" vs. "non-trap" forms. In the "Little Trap" approach:

    .. math::
        d = \sqrt{(\rho\,\sigma\,i\,u - \kappa)^2 + \sigma^2\,(u^2 + i\,u)},\\
        g = \frac{\kappa - \rho\,\sigma\,i\,u \pm d}{\kappa - \rho\,\sigma\,i\,u \mp d},

    where the sign in :math:`\pm` depends on trap. Typically for trap=1 ("Little Trap"):

    .. math::
        g = \frac{\beta - d}{\beta + d}, \quad \text{where } \beta = \kappa - \rho\,\sigma\,i\,u.

    Then,

    .. math::
        B(t,u) = \frac{\beta - d}{\sigma^2}\,\frac{1 - e^{-d\,t}}{1 - g\,e^{-d\,t}},\quad
        A(t,u) = \frac{\kappa\,\theta}{\sigma^2}\,\bigl[(\beta - d)\,t - 2\,\ln\!\bigl(\frac{1 - g\,e^{-d\,t}}{1 - g}\bigr)\bigr].

    The final CF is:

    .. math::
        \phi(u) = \exp\Bigl(i\,u\,[\ln(S_0) + (r - q)\,t]\Bigr) \times
                  \exp\Bigl(A(t,u) + B(t,u)\,v_0\Bigr).

    This function returns :math:`\phi(u)` as a Python callable, so you can evaluate
    :math:`\phi(u)` for complex :math:`u`.

    Parameters
    ----------
    t : float
        Time to maturity (in years).
    spot : float
        Current spot price (S_0).
    r : float
        Risk-free interest rate.
    q : float
        Continuous dividend yield.
    v0 : float
        Initial variance (v(0)).
    kappa : float
        Mean-reversion speed of the variance.
    theta : float
        Long-run variance level.
    sigma : float
        Volatility of volatility.
    rho : float
        Correlation in [-1, 1].
    trap : int, optional
        1 => "Little Trap" form, 0 => Original Heston. Default 1.

    Returns
    -------
    Callable[[complex], complex]
        A function that, when called with a complex u, returns :math:`\phi(u)`.

    Raises
    ------
    ValueError
        If :math:`\rho` not in [-1,1] or if :math:`sigma < 0`.

    References
    ----------
    See above docstring for references to Heston (1993) and subsequent stable
    implementations (Lord & Kahl, Albrecher's "Little Heston Trap").

    Examples
    --------
    >>> cf = heston_cf_hpc(t=1.0, spot=100.0, r=0.05, q=0.02, v0=0.04, 
    ...                    kappa=1.5, theta=0.04, sigma=0.3, rho=-0.7)
    >>> val = cf(1.0+0j)
    >>> abs(val) > 0
    True
    """
    if not (-1.0 <= rho <= 1.0):
        raise ValueError("Correlation rho must be in [-1,1].")
    if sigma < 0.0:
        raise ValueError("Vol of vol (sigma) must be non-negative.")

    i = 1j  # imaginary unit
    lnS = math.log(spot)

    # Precompute some terms
    def phi(u: complex) -> complex:
        # "Little Trap" or standard approach
        #  alpha = kappa * theta
        #  We define common components:
        alpha = kappa * theta
        # Variation in "trap" approach is about sign in the g function:
        beta = kappa - rho * sigma * i * u
        # eqn inside sqrt:
        d_ = cmath.sqrt(
            (rho * sigma * i * u - kappa) ** 2 + sigma**2 * (u**2 + i * u * 0.0 * 2.0)
        )  # we can rewrite if needed

        if trap == 1:
            # "Little Trap" approach
            g = (beta - d_) / (beta + d_)
        else:
            # original Heston approach
            g = (beta + d_) / (beta - d_)

        # Exponent for A(t,u), B(t,u):
        exp_neg_d_t = cmath.exp(-d_ * t)

        # B(t,u)
        B = (beta - d_) / (sigma**2) * ((1.0 - exp_neg_d_t) / (1.0 - g * exp_neg_d_t))

        # A(t,u)
        # Note the "log((1 - g e^{-d t}) / (1 - g))" part
        # We define the stable log approach with cmath.log, mindful of complex logs
        C = (beta - d_) / (sigma**2)
        D = 2.0 * cmath.log((1.0 - g * exp_neg_d_t) / (1.0 - g))
        A = (alpha / (sigma**2)) * ((beta - d_) * t - D)

        # The main exponential
        # ln(S_0) + (r - q)*t => i u ...
        # Then multiply by exp(A + B * v0)
        term1 = i * u * (lnS + (r - q) * t)
        return cmath.exp(term1 + A + B * v0)

    return phi
