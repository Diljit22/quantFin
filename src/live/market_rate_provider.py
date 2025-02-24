#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
market_rate_provider.py
=======================
Provides a function to derive a risk-free rate for a specified maturity and
returns it as an instance of the MarketEnvironment class.

This file uses the FredDataProvider to interpolate the long-term yield for
a given maturity and creates a MarketEnvironment instance with that rate.
It also offers a simple CLI interface.

Usage:
    python market_rate_provider.py --maturity 5.0
"""

import argparse
import logging
from src.live import FredDataProvider
from src.market import MarketEnvironment

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def get_market_environment_by_maturity(maturity: float) -> MarketEnvironment:
    """
    Retrieve the risk-free rate for the specified maturity and return it as a
    MarketEnvironment instance.

    Parameters
    ----------
    maturity : float
        The target maturity in years (e.g., 1.34 or 5.0).

    Returns
    -------
    MarketEnvironment
        An instance of MarketEnvironment with its rate set to the interpolated
        long‑term rate for the given maturity.

    Raises
    ------
    ValueError
        If the rate cannot be retrieved or interpolation fails.
    """
    provider = FredDataProvider()
    rate = provider.get_long_term_rate(maturity)
    return MarketEnvironment(rate=rate)


def main() -> None:
    """
    CLI to retrieve and display a MarketEnvironment instance for a given maturity.

    Usage
    -----
    python market_rate_provider.py --maturity 5.0
    """
    parser = argparse.ArgumentParser(
        description="Retrieve a MarketEnvironment instance for a given maturity."
    )
    parser.add_argument(
        "--maturity",
        type=float,
        required=True,
        help="Target maturity in years (e.g., 5.0).",
    )
    args = parser.parse_args()

    try:
        market_env = get_market_environment_by_maturity(args.maturity)
        print(market_env)
    except Exception as e:
        logging.error("Error retrieving market environment: %s", e)
