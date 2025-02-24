#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
live package
============
This package provides live market data retrieval and context creation for
option pricing. It includes modules for interacting with external APIs
(FRED for interest rates, Polygon for stock/option data) and building
objects like MarketEnvironment, Stock, and EuropeanOption with real(ish) data.

Modules
-------
fred_data_provider
    Defines the FredDataProvider class for fetching Treasury yields and rates from FRED.
market_rate_provider
    Defines get_market_environment_by_maturity, returning a MarketEnvironment with
    an interpolated yield for a specified maturity.
polygon_data_provider
    Defines the PolygonDataProvider class for fetching equities/options data
    from Polygon, including spot price, historical volatility, and
    OSI symbol parsing.
market_option_provider
    Defines functions that combine Polygon data and FRED rates to create Stock, Option,
    and MarketEnvironment objects (e.g., create_full_market_context).
"""

from .fred_data_provider import FredDataProvider
from .market_rate_provider import get_market_environment_by_maturity
from .polygon_data_provider import PolygonDataProvider
from .market_option_provider import (
    create_full_market_context,
    init_european_option_stock_market_env,
)

__all__ = [
    "FredDataProvider",
    "get_market_environment_by_maturity",
    "PolygonDataProvider",
    "create_full_market_context",
    "init_european_option_stock_market_env",
]
