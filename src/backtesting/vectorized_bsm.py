#!/usr/bin/env python3
"""
pricing_functions.py

This module provides example vectorized pricing functions.
For demonstration, we implement the Black‑Scholes pricing formula for European call options.
"""

import numpy as np
from scipy.stats import norm


def black_scholes_call_vectorized(
    strikes: np.ndarray,
    spot: float,
    maturity: float,
    risk_free_rate: float,
    volatility: float,
    dividend: float,
) -> np.ndarray:
    """
    Compute Black‑Scholes call option prices vectorized over strikes.

    Parameters
    ----------
    strikes : np.ndarray
         Array of option strike prices.
    spot : float
         Current spot price of the underlying.
    maturity : float
         Time to maturity in years.
    risk_free_rate : float
         Annual risk‑free rate (as a decimal).
    volatility : float
         Annualized volatility (as a decimal).
    dividend : float
         Continuous dividend yield (as a decimal).

    Returns
    -------
    np.ndarray
         Array of call option prices.
    """
    if maturity <= 0:
        return np.maximum(spot - strikes, 0)

    d1 = (
        np.log(spot / strikes)
        + (risk_free_rate - dividend + 0.5 * volatility**2) * maturity
    ) / (volatility * np.sqrt(maturity))
    d2 = d1 - volatility * np.sqrt(maturity)
    call_prices = spot * np.exp(-dividend * maturity) * norm.cdf(d1) - strikes * np.exp(
        -risk_free_rate * maturity
    ) * norm.cdf(d2)
    return call_prices
