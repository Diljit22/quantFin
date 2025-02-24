#!/usr/bin/env python3
"""
option_pricer.py

This module provides functionality to price a set of options for a given stock at a single time snapshot,
using a vectorized pricing function.

The pricing function should have the signature:
    pricing_function(strikes: np.ndarray, spot: float, maturity: float, risk_free_rate: float, volatility: float, dividend: float) -> np.ndarray

It returns an array of model option prices corresponding to the input strikes.
"""

import numpy as np


def price_options_for_stock(
    stock, market_env, option_chain, maturity, pricing_function
):
    """
    Prices a set of options given a stock and market environment.

    Parameters
    ----------
    stock : Stock
         Instance of Stock containing spot, volatility, dividend, etc.
    market_env : MarketEnvironment
         Instance of MarketEnvironment containing risk‑free rate.
    option_chain : list
         List of dicts with keys 'strike' and 'price' for actual market prices.
    maturity : float
         Time to maturity in years for the options being priced.
    pricing_function : Callable
         A vectorized pricing function as described above.

    Returns
    -------
    list of dict
         Each dict contains 'strike', 'actual_price', and 'model_price'.
    """
    strikes = np.array([entry["strike"] for entry in option_chain])
    spot = stock.spot
    risk_free_rate = market_env.rate
    volatility = stock.volatility
    dividend = stock.dividend
    model_prices = pricing_function(
        strikes, spot, maturity, risk_free_rate, volatility, dividend
    )
    priced_options = []
    for i, strike in enumerate(strikes):
        priced_options.append(
            {
                "strike": strike,
                "actual_price": option_chain[i]["price"],
                "model_price": model_prices[i],
            }
        )
    return priced_options
