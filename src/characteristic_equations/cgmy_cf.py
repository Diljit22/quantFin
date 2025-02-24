#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cgmy_cf.py
==========
Provides the characteristic function for the CGMY (Carr–Geman–Madan–Yor) model,
ensuring the risk-neutral drift correction so that E[S_t] = S_0 e^{(r - q) t}.

References
----------
- Carr, P., Geman, H., Madan, D. B., & Yor, M. (2002). "The Fine Structure of
  Asset Returns: An Empirical Investigation." Journal of Business.
- Madan, D. B., & Yor, M. (2008). "CGMY and beyond."

Usage
-----
    from characteristic_equations.cgmy_cf import cgmy_cf
    cf = cgmy_cf(t=1.0, spot=100.0, r=0.05, q=0.02, C=1.0, G=5.0, M=5.0, Y=0.5)
    val = cf(1.0 + 0j)
    print(val)
"""

import math
import cmath
from typing import Callable

def cgmy_cf(
    t: float,
    spot: float,
    r: float,
    q: float,
    C: float,
    G: float,
    M: float,
    Y: float
) -> Callable[[complex], complex]:
    r"""
    Return the characteristic function for the log of the asset price under the CGMY model.

    The CGMY Lévy process X_t has characteristic exponent:

    .. math::
        \Psi_{CGMY}(u) = C \Gamma(-Y)\bigl[ (M - i u)^Y - M^Y + (G + i u)^Y - G^Y \bigr],

    so that the characteristic function for increments over time t is:

    .. math::
        \phi_{X_t}(u) = \exp\bigl( t \Psi_{CGMY}(u) \bigr).

    We also require a risk-neutral drift correction so that E[e^{X_t}] = e^{(r-q) t}.
    This means we solve:

    .. math::
        e^{\Psi_{CGMY}(-i) \, t} \times e^{\text{driftCorr} \, i(-i) t} = e^{(r-q)t}.

    Implementation
    -------------
    We compute the "pure" CGMY increment exponent, then shift by a driftCorr to achieve
    the risk-neutral condition. Finally:

    .. math::
        X_t = \ln(spot) + driftCorr \times t + (\text{pure CGMY increment}).

    The CF is \phi_X(u) = \exp( i u [ \ln(spot) + driftCorr t ] ) \times \exp( t \Psi_{CGMY}(u) ).

    Parameters
    ----------
    t : float
        Time to maturity.
    spot : float
        Spot price S0.
    r : float
        Risk-free rate.
    q : float
        Dividend yield.
    C : float
        CGMY parameter (C>0).
    G : float
        CGMY parameter, typically > 0.
    M : float
        CGMY parameter, typically > 0.
    Y : float
        CGMY parameter in (0,2) typically.

    Returns
    -------
    Callable[[complex], complex]
        A function phi(u) that gives the CF at complex u.

    Raises
    ------
    ValueError
        If parameters are out of domain or if \Psi_{CGMY}(-i) is invalid.

    Example
    -------
    >>> cf = cgmy_cf(1.0, 100.0, 0.05, 0.02, C=1.0, G=5.0, M=5.0, Y=0.5)
    >>> val = cf(1.0+0j)
    >>> abs(val) > 0
    True
    """

    if C <= 0.0 or G <= 0.0 or M <= 0.0 or not (0 < Y < 2):
        raise ValueError("CGMY parameters out of domain: C>0, G>0, M>0, 0<Y<2 recommended.")

    i = 1j
    lnS = math.log(spot)

    # The "pure" CGMY exponent function, for increment over 1 time unit:
    #   \Psi(u) = C Gamma(-Y)[ (M - i u)^Y - M^Y + (G + i u)^Y - G^Y ].
    # Then for time t => multiply by t.
    def cgmy_exponent(u: complex) -> complex:
        # We define
        c1 = (M - i*u)**Y - (M**Y)
        c2 = (G + i*u)**Y - (G**Y)
        # gamma(-Y) is well-defined if Y<1? We assume Y<2 for the standard model, might be partial.
        gamma_negY = cmath.gamma(-Y)
        return C*gamma_negY*(c1 + c2)

    # Evaluate at -i to find driftCorr.
    # We want e^{ t [ \Psi(-i) + i * driftCorr * (-i) ] } = e^{(r-q) t}.
    # => \Psi(-i) + driftCorr = (r-q).
    val_at_minus_i = cgmy_exponent(-i)
    # we want driftCorr = (r-q) - val_at_minus_i
    # but note that i * driftCorr * (-i) = + driftCorr => so that works out.
    driftCorr = (r - q) - val_at_minus_i.real  # it should be purely real if parameters valid
    # We expect val_at_minus_i to be real if Y is in (0,1)? Actually, for 0<Y<2 might still be real?
    # We check if its imaginary part is negligible:
    if abs(val_at_minus_i.imag) > 1e-10:
        raise ValueError("CGMY exponent at -i is not purely real => check parameters or domain constraints.")

    def phi(u: complex) -> complex:
        # CF = exp( i u [ lnS + driftCorr * t ] ) * exp( t * \Psi(u) )
        main_exp = cmath.exp(i*u*(lnS + driftCorr*t))
        pure_part = cmath.exp(t*cgmy_exponent(u))
        return main_exp*pure_part

    return phi
