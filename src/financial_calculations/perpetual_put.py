#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
perpetual_put.py
================

Provides a function to price a perpetual put option (an American put with no expiration)
using a closed-form solution derived from an associated ODE.

Functions
---------
perpetual_put(S, K, r, vol, q)
    Computes the price of a perpetual put option.
"""

import math


def perpetual_put(S: float, K: float, r: float, vol: float, q: float) -> float:
    """
    Compute the price of a perpetual put option.

    The pricing is based on solving the ODE that arises from the optimal stopping
    problem of an American put with no expiration. The closed-form solution is given by:

        Price = (K / (1 - beta)) * (((beta - 1) / beta) * (S / K))**beta

    where beta is the negative root of the quadratic equation derived from the model:

        vol^2 * beta^2 + 2 * (r - q - vol^2 / 2) * beta - 2 * r = 0

    In this implementation, beta is computed as:

        beta = - (b + sqrt(discriminant)) / vol_sq

    with:
        vol_sq = vol**2,
        b = r - q - vol_sq/2,
        discriminant = b**2 + 2 * r * vol_sq.

    Parameters
    ----------
    S : float
        Current stock price.
    K : float
        Strike price of the option.
    r : float
        Annualized risk-free interest rate (continuously compounded).
    vol : float
        Volatility of the stock.
    q : float
        Continuous dividend yield.

    Returns
    -------
    float
        The price of the perpetual put option.

    Raises
    ------
    ValueError
        If r is zero (the model requires r != 0).

    Examples
    --------
    >>> perpetual_put(S=150, K=100, r=0.08, vol=0.2, q=0.005)
    1.8344292693352158
    """
    if r == 0:
        raise ValueError("Risk-free rate r must be nonzero.")

    vol_sq = vol**2
    b = r - q - vol_sq / 2
    discriminant = b**2 + 2 * r * vol_sq
    beta = -(b + math.sqrt(discriminant)) / vol_sq

    # Compute the solution to the ODE.
    mnRoot = beta - 1
    mul = -K / mnRoot
    base = (mnRoot / beta) * (S / K)
    price = mul * (base**beta)

    return price
