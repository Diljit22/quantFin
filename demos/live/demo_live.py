#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
demo_live.py
============
A demonstration of the live package usage:
  - FredDataProvider for short-term and long-term rates.
  - get_market_environment_by_maturity for an interpolated MarketEnvironment.
  - create_full_market_context for building a Stock and EuropeanOption.

"""

import sys
from src.live import (
    FredDataProvider,
    get_market_environment_by_maturity,
    create_full_market_context,
)


def main():
    print("=== Demo: FredDataProvider ===")
    fred_provider = FredDataProvider()
    fred_provider.main_cli()  # You can comment this out if you don't want the CLI.

    short_rate = fred_provider.get_short_term_rate(series_id="DTB3")
    print(f"Short-Term (3-mo T-bill) Rate => {short_rate:.4f}")

    long_rate_1y2m = fred_provider.get_long_term_rate(maturity=1.2)
    print(f"Long-Term Rate for 1.2-year maturity => {long_rate_1y2m:.4f}")

    print("\n=== Demo: get_market_environment_by_maturity ===")
    market_env = get_market_environment_by_maturity(1.2)
    print(f"Market Environment => {market_env}\n")

    print("=== Demo: create_full_market_context ===")
    try:
        # Example symbol: 'O:SPY251219C00650000
        env, stock, option = create_full_market_context("O:SPY251219C00650000")
        print(f"MarketEnvironment => {env}")
        print(f"Stock => {stock}")
        print(f"EuropeanOption => {option}")
    except SystemExit:
        print(
            "Option is expired or data retrieval failed. (Using dummy or invalid keys?)"
        )
        sys.exit(1)
    except Exception as e:
        print(f"Error in create_full_market_context: {e}")


if __name__ == "__main__":
    main()
