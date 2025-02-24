#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
variance_gamma_cf.py
====================
Provides the characteristic function for the Variance Gamma (VG) model.

Under one standard parameterization, the log-price X_t has increments that follow
a variance gamma process with parameters (sigma, theta, nu). We incorporate a
risk-neutral drift correction to ensure E[e^{X_t}] = e^{(r - q) t}.

References
----------
- Madan, D.B., Carr, P., Chang, E.C. (1998). "The Variance Gamma Process and
  Option Pricing." European Finance Review.
- Madan, D.B., Seneta, E. (1990). "The VG model for share market returns."

Usage
-----
    from characteristic_equations.variance_gamma_cf import variance_gamma_cf
    cf = variance_gamma_cf(t, spot, r, q, sigma, theta, nu)
    val = cf(1.0+0j)
    print(val)
"""

import math
import cmath
from typing import Callable


def variance_gamma_cf(
    t: float, spot: float, r: float, q: float, sigma: float, theta: float, nu: float
) -> Callable[[complex], complex]:
    r"""
    Return the characteristic function for the log of the asset price under the
    Variance Gamma model (with parameters sigma, theta, nu).

    The log-price X_t = ln(S_t) is modeled as:
       X_t = ln(spot) + (r - q - driftCorr)*t + a VarianceGamma(...) increment,
    where driftCorr ensures E[e^{X_t}] = e^{(r - q) t}.

    The standard VG characteristic function for increment Y_t is:
    .. math::
        \phi_{VG}(u) = \Bigl(1 - i\,\theta\,\nu\,u + 0.5\,\sigma^2\,\nu\,u^2\Bigr)^{-t/\nu},

    and we incorporate the drift correction to shift the exponent.

    Parameters
    ----------
    t : float
        Time to maturity (years).
    spot : float
        Current asset price (S_0).
    r : float
        Risk-free rate.
    q : float
        Continuous dividend yield.
    sigma : float
        Vol parameter in the VG subordinator approach.
    theta : float
        Drift of the underlying Brownian in the VG process.
    nu : float
        The 'time' or 'Gamma' scale parameter. (1/nu is the shape param of the subordinator)

    Returns
    -------
    Callable[[complex], complex]
        A function phi(u) that computes the CF at complex u.

    Notes
    -----
    The drift correction ensures the risk-neutral condition E[S_t] = S_0 e^{(r-q) t}.
    Typically, we define:
        kappa = -\frac{1}{\nu} \ln(1 - \theta\,\nu - 0.5\,\sigma^2\,\nu)
    => so that E[e^{X_t}] = e^{(r-q) t} if X_t is purely the VG part plus an adjusted drift.

    One direct approach is to store 'driftCorr = \ln(\ldots ) / \nu' or so. Here, we do
    a simpler "full formula" approach.

    Example
    -------
    >>> cf = variance_gamma_cf(1.0, 100.0, 0.05, 0.02, 0.2, -0.1, 0.3)
    >>> val = cf(1.0+0j)
    >>> abs(val) > 0
    True
    """
    # 1) Compute the log(S_0) part
    lnS = math.log(spot)

    # 2) We define a standard 'kappa' or 'drift correction' so that E[e^{X_t}] = e^{(r-q)t}.
    #    The characteristic function for increment Y ~ VG(sigma, nu, theta) is:
    #       phi_Y(u) = (1 - i theta nu u + 0.5 sigma^2 nu u^2)^(-1/nu)
    #    Over time t => exponent is t/nu.
    # Then X_t = Y_t + (some drift)*t. The drift is chosen to match risk-neutral.
    # common approach:
    # driftCorr = (r - q)* ...
    # Actually, in many references:
    #   kappa = - 1/nu * ln(1 - theta nu - 0.5 sigma^2 nu).
    # The net drift => (r - q) - something = ?

    # Let's define the increment CF for Y_t over time t:
    def increment_cf(u: complex) -> complex:
        """
        The CF of Y_t (the pure VG increment) over [0,t].
        """
        i = 1j
        # base term inside parentheses
        z = 1.0 - i * theta * nu * u + 0.5 * sigma * sigma * nu * (u**2)
        # exponent
        power = -(t / nu)
        return z ** (power)

    # We define log offset = i u [ lnS + (r - q)*t - (some correction)*t ]
    # Actually, we can do:
    # Let 'kappa' = 1/nu * ln(1 - theta nu - 0.5 sigma^2 nu ) negative => or we define the sign carefully.
    # We'll do a more direct approach: E[e^{Y_t}] = (some expression). Then we require e^{driftCorr * t} * E[e^{Y_t}] = e^{(r-q) t}.
    # => driftCorr = (r - q) - 1/t ln(E[e^{Y_t}]).
    # E[e^{Y_t}] = increment_cf(- i ), i.e. phi_Y(- i ).
    i = 1j
    # Evaluate increment_cf(-i):
    val_at_minus_i = increment_cf(-i)
    # We want e^{driftCorr * t} * val_at_minus_i = e^{(r-q) t}
    # => driftCorr = (r - q) - (1/t) ln(val_at_minus_i)
    # but val_at_minus_i might be complex. The VG at -i should be real if parameters are valid. We'll assume it is.
    # let's define
    if abs(val_at_minus_i.imag) > 1e-12:
        raise ValueError("VG CF at -i is not purely real, check parameters.")
    temp = val_at_minus_i.real
    if temp <= 0:
        raise ValueError(
            "VG CF at -i <= 0 => invalid parameters, or drift correction fails."
        )
    driftCorr = (
        (r - q) - (1.0 / t) * math.log(temp) if t > 1e-16 else (r - q)
    )  # if t=0 ?

    def phi(u: complex) -> complex:
        # Combine the logS + driftCorr with the pure VG increment CF
        # => \phi_X(u) = exp(i u [ lnS + driftCorr * t ]) * \phi_Y(u)
        # plus we might incorporate (r - q)* t in there, but we already folded that into driftCorr
        i = 1j
        main_exp = cmath.exp(i * u * (lnS + driftCorr * t))
        return main_exp * increment_cf(u)

    return phi
