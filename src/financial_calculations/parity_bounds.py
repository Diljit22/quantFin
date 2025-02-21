"""
parity_bounds.py
================

This file provides a collection of utility functions related to the put-call
relationship and option price bounds for European (and dividend-less American) options.

Functions
---------
- put_call_parity(option_price, S, K, r, T, q=0.0, price_call=False)
    Computes the complementary option price via put-call parity.
- put_call_bound(option_price, S, K, r, T, bound_call=False)
    Computes naive lower/upper bounds for call or put options.
- lower_bound_rate(call_price, put_price, S, K, T)
    Computes a lower bound on the risk-free rate using the put-call inequality.

"""

import math
import numpy as np
from typing import Union, Tuple


def put_call_parity(
    option_price: float,
    S: float,
    K: float,
    r: float,
    T: float,
    q: Union[float, None] = 0.0,
    price_call: bool = False,
) -> float:
    """
    Compute the complementary European option price using put-call parity.

    The put-call parity formula is:
        C - P = S * exp(-q*T) - K * exp(-r*T)

    Parameters
    ----------
    option_price : float
        The known option price. If `price_call` is False, this is the put price;
        if True, this is the call price.
    S : float
        Current underlying spot price.
    K : float
        Strike price.
    r : float
        Annualized, continuously compounded risk-free rate.
    T : float
        Time to maturity in years.
    q : float or None, optional
        Continuous dividend yield (default is 0.0).
    price_call : bool, default False
        If True, `option_price` is interpreted as a call price (and the computed value
        will be the corresponding put price). Otherwise, it is the put price (and the
        function returns the call price).

    Returns
    -------
    float
        The computed price of the complementary option.

    Examples
    --------
    >>> put_call_parity(6.71, S=100, K=110, r=0.08, T=0.5, q=0.01, price_call=False)
    0.5244096125126907
    """
    if q is None:
        q = 0.0

    discounted_strike = K * math.exp(-r * T)
    discounted_spot = S * math.exp(-q * T)
    parity_diff = discounted_spot - discounted_strike

    if price_call:
        # Given call price, compute put price.
        return option_price - parity_diff
    else:
        # Given put price, compute call price.
        return option_price + parity_diff


def put_call_bound(opPr, S, K, r, T, bound_call=False):
    """
    Compute lower and upper bounds for a European/American option using
    put-call inequalities, incorporating a continuous dividend yield.

    For a call on a dividend-paying asset:
      Lower bound: max(0, S * exp(-q*T) - K * exp(-r*T))
      Upper bound: S * exp(-q*T)
    For a put:
      Lower bound: max(0, K * exp(-r*T) - S * exp(-q*T))
      Upper bound: K * exp(-r*T)

    Parameters
    ----------
    S : float
        Underlying spot price.
    K : float
        Strike price.
    r : float
        Risk-free rate (continuously compounded).
    T : float
        Time to maturity in years.
    q : float, optional
        Continuous dividend yield (default is 0.0).
    bound_call : bool, default False
        If True, compute bounds for a call; otherwise, for a put.

    Returns
    -------
    (float, float)
        A tuple (lower_bound, upper_bound).

    Examples
    --------
    >>> lb, ub = put_call_bound(S=36.0, K=37.0, r=0.055, T=0.5, q=0.01, bound_call=False)
    >>> round(lb, 4), round(ub, 4)
    (0.1759140035412372, 35.9963632544778)
    >>> lb, ub = put_call_bound(S=36.0, K=37.0, r=0.055, T=0.5, q=0.01, bound_call=True)
    >>> round(lb, 4), round(ub, 4)
    (0.0, 35.82044925093656)
    """
    adjK = np.exp(-r * T) * K
    mxDiff = S - adjK
    mnDiff = S - K

    lower = mnDiff + opPr if bound_call else opPr - mxDiff
    upper = mxDiff + opPr if bound_call else opPr - mnDiff

    bounds = (max(lower, 0), upper)
    return bounds


def lower_bound_rate(
    call_price: float, put_price: float, S: float, K: float, T: float
) -> float:
    """
    Compute a lower bound on the risk-free rate using the put-call inequality:
        C - P <= S - K * exp(-r*T)

    Rearranging, we get:
        r >= - (1/T) * ln((S - C + P) / K)

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
        Time to maturity in years.

    Returns
    -------
    float
        Lower bound on the risk-free rate.

    Raises
    ------
    ValueError
        If S - call_price + put_price is not positive (making the logarithm undefined).

    Examples
    --------
    >>> lower_bound_rate(0.5287, 6.7143, 100, 110, 0.5)
    0.07058371879701723
    """
    val = S - call_price + put_price
    if val <= 0.0:
        raise ValueError("S - call_price + put_price must be > 0 to compute logarithm.")
    return -math.log(val / K) / T
