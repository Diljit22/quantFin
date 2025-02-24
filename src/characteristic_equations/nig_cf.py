#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
nig_cf.py
=========
Provides the characteristic function for the Normal Inverse Gaussian (NIG) model,
with a drift correction to ensure E[S_t] = S_0 e^{(r-q)t} under the risk-neutral measure.

References
----------
- Barndorff-Nielsen, O. E. (1997). "Processes of normal inverse Gaussian type."
  Finance and Stochastics.
- Rydberg, T. H. (1999). "Generalized hyperbolic and hyperbolic distributions."
  Mathematical Finance.

Usage
-----
    from characteristic_equations.nig_cf import nig_cf
    cf = nig_cf(t=1.0, spot=100.0, r=0.05, q=0.02, alpha=10.0, beta=-2.0, delta=0.3)
    val = cf(1.0 + 0j)
    print(val)
"""

import math
import cmath
from typing import Callable

def nig_cf(
    t: float,
    spot: float,
    r: float,
    q: float,
    alpha: float,
    beta: float,
    delta: float
) -> Callable[[complex], complex]:
    r"""
    Returns the characteristic function for the log of the asset price under an NIG
    (Normal Inverse Gaussian) process with parameters \(\alpha,\beta,\delta\).

    The NIG characteristic exponent for a time increment \(\Delta t = 1\) is often
    written as:

    .. math::
        \psi_{NIG}(u) = - \delta \left(
          \sqrt{\alpha^2 - ( \beta + i u )^2 }
          - \sqrt{\alpha^2 - \beta^2}
        \right).

    Over time t, the exponent is multiplied by t. We incorporate a drift correction so
    that E[e^{X_t}] = e^{(r-q)t}. Then:

    .. math::
       X_t = \ln(spot) + \text{driftCorr} * t + (NIG increment)...

    and

    .. math::
       \phi_X(u) = \exp\bigl(i u [\ln(spot) + \text{driftCorr} * t]\bigr)
                   \exp\bigl(t \,\psi_{NIG}(u)\bigr).

    The driftCorr is found by requiring \phi_X(-i) = e^{(r-q) t}.

    Parameters
    ----------
    t : float
        Time to maturity (years).
    spot : float
        Spot price S0.
    r : float
        Risk-free rate.
    q : float
        Dividend yield.
    alpha : float
        NIG parameter alpha > 0.
    beta : float
        NIG parameter |beta| < alpha for the process to be well-defined.
    delta : float
        NIG scale parameter > 0.

    Returns
    -------
    Callable[[complex], complex]
        The characteristic function phi(u).

    Raises
    ------
    ValueError
        If domain constraints are violated or if \psi_{NIG}(-i) is complex in an unexpected way.

    Example
    -------
    >>> cf = nig_cf(1.0, 100.0, 0.05, 0.02, alpha=10.0, beta=-2.0, delta=0.3)
    >>> val = cf(1.0+0j)
    >>> abs(val) > 0
    True
    """

    if alpha <= 0.0 or delta <= 0.0:
        raise ValueError("alpha>0 and delta>0 are required for NIG.")
    if abs(beta) >= alpha:
        raise ValueError("Must have |beta| < alpha for a valid NIG process.")

    i = 1j
    lnS = math.log(spot)

    def psi_nig(u: complex) -> complex:
        # \psi_{NIG}(u) = - delta [ sqrt(alpha^2 - (beta + i u)^2 ) - sqrt(alpha^2 - beta^2 ] ]
        # Typically real domain issues if alpha^2 < (beta + i u)^2 but let's trust the principle branch.
        c1 = alpha*alpha - (beta + i*u)**2
        c2 = alpha*alpha - beta*beta
        val = -delta*(cmath.sqrt(c1) - math.sqrt(c2))  # c2 is real +ve if |beta|<alpha
        return val

    # Evaluate at -i for drift correction
    val_minus_i = psi_nig(-i)
    # We want: e^{ t [ val_minus_i + i * driftCorr * (-i) ] } = e^{(r-q) t }
    # => val_minus_i + driftCorr = (r - q)
    # => driftCorr = (r - q) - val_minus_i
    if abs(val_minus_i.imag) > 1e-12:
        raise ValueError("psi_nig(-i) is not purely real => invalid NIG parameters?")

    driftCorr = (r - q) - val_minus_i.real

    def phi(u: complex) -> complex:
        # CF = exp(i u [ lnS + driftCorr t ]) * exp( t * psi_nig(u) )
        main_exp = cmath.exp(i*u*(lnS + driftCorr*t))
        pure_part = cmath.exp(t*psi_nig(u))
        return main_exp * pure_part

    return phi
