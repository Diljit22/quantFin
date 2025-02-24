#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
market_option_provider.py
=========================
Provides functions to create market-related objects (MarketEnvironment,
Stock, EuropeanOption) for a given option symbol or a dictionary of parameters.

Features
--------
1) create_full_market_context(option_symbol):
   - Uses PolygonDataProvider to parse and fetch option data (spot, hist vol, etc.).
   - Computes time-to-expiry from the option's expiration date.
   - Fetches a risk-free rate from FRED (via get_market_environment_by_maturity).
   - Creates and returns a MarketEnvironment, Stock, and EuropeanOption instance.

2) init_european_option_stock_market_env(params):
   - Given a dict of parameters, directly initializes MarketEnvironment, Stock,
   and EuropeanOption without external API calls.

Usage
-----
    from live.market_option_provider import create_full_market_context,
    init_european_option_stock_market_env

    # Example:
    market_env, stock, option = create_full_market_context("O:SPY251219C00650000")

    # Or:
    params = {
        "spot": 100.0,
        "volatility": 0.20,
        "dividend": 0.01,
        "symbol": "AAPL",
        "strike": 100.0,
        "maturity": 1.0,
        "is_call": True,
        "rate": 0.05,
    }
    market_env, stock, option = init_european_option_stock_market_env(params)
"""

import sys
import logging
from datetime import datetime

from src.live import PolygonDataProvider
from src.live import get_market_environment_by_maturity
from src.underlyings import Stock
from src.instruments import EuropeanOption
from src.market import MarketEnvironment

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def create_full_market_context(option_symbol: str):
    """
    Creates a MarketEnvironment, Stock, and EuropeanOption instance.

    1) Fetch option data from PolygonDataProvider
    2) Compute time-to-expiry from the option's expiration date (in years).
    3) Retrieve a risk-free rate from FRED using get_market_environment_by_maturity().
    4) Initialize a Stock instance.
    5) Initialize a EuropeanOption with the strike, maturity, and is_call.

    Parameters
    ----------
    option_symbol : str
        OSI-formatted option symbol (e.g. "O:SPY251219C00650000").

    Returns
    -------
    (MarketEnvironment, Stock, EuropeanOption)
        A tuple containing:
          - MarketEnvironment with rate from FRED,
          - Stock with spot, volatility, etc. from Polygon,
          - EuropeanOption reflecting the parsed strike, expiration, and call/put.

    Raises
    ------
    SystemExit
        If the option has expired or data retrieval fails.
    """
    # 1) Fetch data from Polygon
    poly_provider = PolygonDataProvider()
    opt_data = poly_provider.get_option_data(option_symbol)
    # e.g. opt_data = {
    #   'underlying': 'SPY',
    #   'expiration': datetime(...),
    #   'option_type': 'C',
    #   'strike': 650.0,
    #   'spot_price': 640.0,
    #   'historical_volatility': 22.5,  # percent
    #   'dividend_info': 'Not Implemented'
    # }

    # 2) Compute time-to-expiry in years
    expiration_date = opt_data["expiration"]
    now = datetime.now()
    days_to_expiry = (expiration_date - now).days
    if days_to_expiry <= 0:
        logging.error(
            "Option expiration has past or is today. Invalid for new option context."
        )
        sys.exit(1)

    maturity = days_to_expiry / 365.0

    # 3) Retrieve risk-free rate from FRED
    market_env = get_market_environment_by_maturity(maturity)

    # 4) Build the Stock
    hist_vol_decimal = opt_data["historical_volatility"] / 100.0
    stock = Stock(
        spot=opt_data["spot_price"],
        volatility=hist_vol_decimal,
        dividend=0.0,  # placeholder or advanced logic if you have dividend data
        symbol=opt_data["underlying"],
    )

    # 5) Build the EuropeanOption
    is_call = opt_data["option_type"] == "C"
    option = EuropeanOption(
        strike=opt_data["strike"], maturity=maturity, is_call=is_call
    )

    logging.info(
        "Created MarketEnvironment(rate=%.4f), Stock(%s), and EuropeanOption(strike=%.2f, is_call=%s, maturity=%.3f)",
        market_env.rate,
        stock.symbol,
        option.strike,
        option.is_call,
        option.maturity,
    )

    return market_env, stock, option


def init_european_option_stock_market_env(params: dict):
    """
    Initialize a MarketEnvironment, Stock, and EuropeanOption instance from dict.

    This function does NOT fetch any data from external APIs. It simply uses
    the provided arguments to create the objects.

    Expected keys in `params`:
        spot (float): Underlying stock spot price
        volatility (float): Annualized volatility as a decimal (e.g., 0.20 for 20%)
        dividend (float): Continuous dividend yield or approximate
        symbol (str): Ticker symbol for the Stock
        strike (float): Option strike price
        maturity (float): Time to expiration in years
        is_call (bool): True for a call, False for a put
        rate (float): Risk-free rate (decimal, e.g., 0.05 for 5%)

    Parameters
    ----------
    params : dict
        A dictionary containing stock and option parameters.

    Returns
    -------
    (MarketEnvironment, Stock, EuropeanOption)
        Instances of the classes reflecting the provided parameters.

    Raises
    ------
    KeyError
        If required keys are missing.
    ValueError
        If any parameter is invalid (e.g., negative maturity, negative spot, etc.).
    """
    required_keys = [
        "spot",
        "volatility",
        "dividend",
        "symbol",
        "strike",
        "maturity",
        "is_call",
        "rate",
    ]
    missing = [k for k in required_keys if k not in params]
    if missing:
        raise KeyError(f"Missing required keys: {missing}")

    # Basic validation
    spot = params["spot"]
    vol = params["volatility"]
    div = params["dividend"]
    sym = params["symbol"]
    strike = params["strike"]
    mat = params["maturity"]
    is_call = params["is_call"]
    rate = params["rate"]

    if spot <= 0:
        raise ValueError("Spot price must be positive.")
    if vol < 0:
        raise ValueError("Volatility must be non-negative.")
    if mat <= 0:
        raise ValueError("Maturity must be positive.")
    # Additional checks can go here.

    market_env = MarketEnvironment(rate=rate)
    stock = Stock(spot=spot, volatility=vol, dividend=div, symbol=sym)
    option = EuropeanOption(strike=strike, maturity=mat, is_call=is_call)

    logging.info(
        "Created MarketEnvironment(rate=%.4f), Stock(%s), and EuropeanOption(strike=%.2f, is_call=%s, maturity=%.3f)",
        rate,
        sym,
        strike,
        is_call,
        mat,
    )

    return market_env, stock, option


def main_cli() -> None:
    """
    Optional mini-CLI to demonstrate create_full_market_context usage.

    Usage
    -----
    python market_option_provider.py --option-symbol O:SPY251219C00650000
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Create a MarketEnvironment, Stock, and EuropeanOption from symbol."
    )
    parser.add_argument(
        "--option-symbol",
        type=str,
        required=True,
        help="OSI-formatted option symbol (e.g. 'O:SPY251219C00650000').",
    )
    args = parser.parse_args()

    market_env, stock, option = create_full_market_context(args.option_symbol)
    print("\n--- Created MarketEnvironment ---")
    print(market_env)
    print("\n--- Created Stock ---")
    print(stock)
    print("\n--- Created EuropeanOption ---")
    print(option)
    print()
