#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
bates_cf_hpc.py
===============
Provides the HPC ("Little Trap") style characteristic function for the Bates model.

The Bates model combines:
  - Heston stochastic volatility dynamics,
  - Merton-type jumps in the asset price.

References
----------
- Bates, D. S. (1996). “Jumps and Stochastic Volatility: Exchange Rate Processes
  Implicit in Deutsche Mark Options.” The Review of Financial Studies, 9(1).
- Heston, S. (1993). A Closed-Form Solution for Options with Stochastic Volatility.
- Lord, R. and Kahl, C. (2010). "Complex Logarithms in Heston-like Models."

Usage
-----
    from characteristic_equations.bates_cf import bates_cf_hpc
    cf = bates_cf_hpc(
        t=1.0,
        spot=100.0,
        r=0.05,
        q=0.02,
        v0=0.04,
        kappa=1.5,
        theta=0.04,
        sigma=0.3,
        rho=-0.7,
        jump_intensity=0.1,
        muJ=-0.1,
        sigmaJ=0.2,
        trap=1
    )
    val = cf(1.0+0j)
    print(val)
"""

import math
import cmath
from typing import Callable


def bates_cf_hpc(
    t: float,
    spot: float,
    r: float,
    q: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    jump_intensity: float,
    muJ: float,
    sigmaJ: float,
    trap: int = 1,
) -> Callable[[complex], complex]:
    r"""
    Compute the HPC ("Little Trap") style characteristic function for the Bates model.

    The Bates model extends Heston by adding Poisson jumps (as in Merton).
    Under risk-neutral measure, the log-price's characteristic function is:

    .. math::
        \phi_{Bates}(u) = \exp\Bigl(iu(\ln(S_0) + (r - q - \lambda\,\kappa_J)\,t)\Bigr)
                          \,\exp\bigl(A(t,u) + B(t,u)\,v_0\bigr)
                          \,\exp\Bigl(\lambda\,\frac{1 - e^{-d\,t}}{1 - g\,e^{-d\,t}}\times\dots\Bigr)

    Actually, an easier approach is to take Heston's CF and multiply by a jump factor:

    .. math::
        \phi_{\text{Bates}}(u) = \phi_{\text{Heston}}\Bigl(u; \text{adjusted drift}\Bigr)
                                \times \exp\bigl(\lambda\,t\,(\exp(i\,u\,\mu_J - 0.5\,\sigma_J^2\,u^2) - 1)\bigr),

    where the Heston CF is computed with (r - q) replaced by (r - q - \lambda\,\kappa_J),
    and :math:`\kappa_J = \exp(\mu_J + 0.5\,\sigma_J^2) - 1`, the jump compensator.

    Parameters
    ----------
    t : float
        Time to maturity (in years).
    spot : float
        Spot price S0.
    r : float
        Risk-free interest rate.
    q : float
        Continuous dividend yield.
    v0 : float
        Initial variance.
    kappa : float
        Mean-reversion speed for variance.
    theta : float
        Long-run variance level.
    sigma : float
        Volatility of volatility.
    rho : float
        Correlation in [-1, 1].
    jump_intensity : float
        Poisson jump intensity, :math:`\lambda`.
    muJ : float
        Mean jump size (in log-space).
    sigmaJ : float
        Volatility of jump sizes in log-space.
    trap : int, optional
        1 => "Little Trap" HPC form, 0 => original Heston form.

    Returns
    -------
    Callable[[complex], complex]
        A function that, given a complex u, returns the characteristic function :math:`\phi_{\text{Bates}}(u)`.

    Notes
    -----
    Let :math:`\kappa_J = \exp(\mu_J + 0.5\,\sigma_J^2) - 1`.
    Then the drift adjustment is :math:`(r - q - \lambda\,\kappa_J)` in place of (r - q).

    The Heston CF part is computed HPC style, and we multiply by Merton's jump factor:
    .. math::
        \exp\bigl(\lambda\,t\,[\exp(i\,u\,\mu_J - 0.5\,\sigma_J^2\,u^2) - 1]\bigr).

    See:
      - Bates (1996) for the original derivation.
      - Heston references, plus "Little Trap" for HPC stability.

    Example
    -------
    >>> cf = bates_cf_hpc(1.0, 100.0, 0.05, 0.02, 0.04, 1.5, 0.04, 0.3, -0.7, 0.1, -0.1, 0.2)
    >>> val = cf(1.0+0j)
    >>> abs(val) > 0
    True
    """
    if not (-1.0 <= rho <= 1.0):
        raise ValueError("rho must be in [-1, 1].")
    if sigma < 0.0:
        raise ValueError("Vol of vol (sigma) must be >= 0.")
    if jump_intensity < 0.0:
        raise ValueError("Jump intensity must be >= 0.")

    i = 1j
    lnS = math.log(spot)
    # compute jump compensator:
    kappaJ = math.exp(muJ + 0.5 * sigmaJ * sigmaJ) - 1.0
    # Adjusted drift: (r - q - lambda*kappaJ)
    adj_rq = (r - q) - jump_intensity * kappaJ

    def cf_heston_part(u: complex) -> complex:
        """
        HPC style Heston CF with drift replaced by adj_rq.
        """
        beta = kappa - rho * sigma * i * u
        d_ = cmath.sqrt(
            (rho * sigma * i * u - kappa) ** 2
            + sigma * sigma * (u * u + i * u * 0.0 * 2.0)
        )
        if trap == 1:
            g = (beta - d_) / (beta + d_)
        else:
            g = (beta + d_) / (beta - d_)

        exp_neg_dt = cmath.exp(-d_ * t)
        B = (beta - d_) / (sigma * sigma) * (1.0 - exp_neg_dt) / (1.0 - g * exp_neg_dt)
        # A(t,u)
        log_part = cmath.log((1.0 - g * exp_neg_dt) / (1.0 - g))
        A = (kappa * theta / (sigma * sigma)) * ((beta - d_) * t - 2.0 * log_part)

        # main exponent
        drift_term = i * u * (lnS + adj_rq * t)
        return cmath.exp(drift_term + A + B * v0)

    def cf_merton_jump(u: complex) -> complex:
        """
        Merton jump factor: exp( lambda * t * [exp(i*u*muJ - 0.5*sigmaJ^2 * u^2) - 1] ).
        """
        jump_expo = cmath.exp(i * u * muJ - 0.5 * sigmaJ * sigmaJ * (u**2))
        return cmath.exp(jump_intensity * t * (jump_expo - 1.0))

    def phi(u: complex) -> complex:
        # Combine Heston CF * Merton jump factor
        return cf_heston_part(u) * cf_merton_jump(u)

    return phi
