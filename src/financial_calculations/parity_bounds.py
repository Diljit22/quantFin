#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
parity_bounds.py
================

This file provides utility functions related to the put-call relationship and
option price bounds for European (and dividend-less American) options.

Functions
---------
put_call_parity(option_price, S, K, r, T, q=0.0, price_call=False)
    Computes the complementary option price via put-call parity.
put_call_bound(opPr, S, K, r, T, bound_call=False)
    Computes naive lower/upper bounds for call or put options.
lower_bound_rate(call_price, put_price, S, K, T)
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
        The known option price. If `price_call` is False, this is interpreted as
        the put price; if True, as the call price.
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
        If True, `option_price` is a call price (and the computed value will be the
        corresponding put price). Otherwise, it is the put price (and the function
        returns the call price).

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
        # Given call price, compute the corresponding put price.
        return option_price - parity_diff
    else:
        # Given put price, compute the corresponding call price.
        return option_price + parity_diff


def put_call_bound(
    option_price: float,
    S: float,
    K: float,
    r: float,
    T: float,
    bound_call: bool = False,
) -> Tuple[float, float]:
    """
    Compute lower and upper bounds for a European or dividend-less American option using
    put-call inequalities.

    Parameters
    ----------
    option_price : float
        The known option price.
    S : float
        Underlying spot price.
    K : float
        Strike price.
    r : float
        Risk-free rate (continuously compounded).
    T : float
        Time to maturity in years.
    bound_call : bool, default False
        If True, compute bounds for a call; otherwise, for a put.

    Returns
    -------
    tuple of float
        A tuple (lower_bound, upper_bound).

    """
    adjK = np.exp(-r * T) * K
    mxDiff = S - adjK
    mnDiff = S - K

    lower = mnDiff + option_price if bound_call else option_price - mxDiff
    upper = mxDiff + option_price if bound_call else option_price - mnDiff

    return (max(lower, 0), upper)


def lower_bound_rate(
    call_price: float, put_price: float, S: float, K: float, T: float
) -> float:
    """
    Compute a lower bound on the risk-free rate using the put-call inequality:
        C - P <= S - K * exp(-r*T)

    Rearranging, the lower bound on r is given by:
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
        If (S - call_price + put_price) is not positive, making the logarithm undefined.

    Examples
    --------
    >>> lower_bound_rate(0.5287, 6.7143, 100, 110, 0.5)
    0.07058371879701723
    """
    val = S - call_price + put_price
    if val <= 0.0:
        raise ValueError("S - call_price + put_price must be > 0 to compute logarithm.")
    return -math.log(val / K) / T
