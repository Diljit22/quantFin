#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_market_option_provider.py
==============================
Pytest suite for the market_option_provider module.

This module tests:
  - create_full_market_context
  - init_european_option_stock_market_env
"""

import math
import pytest
import sys
from datetime import datetime, timedelta
from src.live.market_option_provider import (
    create_full_market_context,
    init_european_option_stock_market_env,
)
from src.underlyings.stock import Stock
from src.instruments.european_option import EuropeanOption
from src.market.market_environment import MarketEnvironment


def dummy_polygon_option_data(self, option_symbol):
    """
    Return a simulated dictionary from get_option_data.
    """
    return {
        "underlying": "TSLA",
        "expiration": datetime.now() + timedelta(days=90),  # 3 months out
        "option_type": "C",
        "strike": 650.0,
        "spot_price": 640.0,
        "historical_volatility": 25.0,
        "dividend_info": "Not Implemented",
    }


def dummy_get_market_env_by_mat(maturity: float) -> MarketEnvironment:
    """
    Return a MarketEnvironment with a rate= maturity*0.01
    purely for test demonstration.
    """
    return MarketEnvironment(rate=maturity * 0.01)


def test_create_full_market_context(monkeypatch):
    """
    Test create_full_market_context by patching out polygon/fred calls.
    """
    from src.live.polygon_data_provider import PolygonDataProvider
    from src.live.market_rate_provider import get_market_environment_by_maturity

    monkeypatch.setattr(
        PolygonDataProvider, "get_option_data", dummy_polygon_option_data
    )
    monkeypatch.setattr(
        "src.live.market_option_provider.get_market_environment_by_maturity",
        dummy_get_market_env_by_mat,
    )

    env, stock, option = create_full_market_context("O:TSLA250123C00650000")

    assert isinstance(env, MarketEnvironment)
    assert isinstance(stock, Stock)
    assert isinstance(option, EuropeanOption)
    # Check if the rate matches the dummy method => maturity ~ 0.246575 => rate=0.002465 or something
    # We won't test the exact number, just that it's >0
    assert env.rate > 0
    assert math.isclose(stock.spot, 640.0, rel_tol=1e-3)
    assert math.isclose(stock.volatility, 0.25, rel_tol=1e-3)
    assert math.isclose(option.strike, 650.0, rel_tol=1e-3)
    assert option.is_call is True


def test_create_full_market_context_expired(monkeypatch):
    """
    If the option is expired or expires today, it should sys.exit(1).
    """
    from src.live.polygon_data_provider import PolygonDataProvider

    monkeypatch.setattr(
        PolygonDataProvider,
        "get_option_data",
        lambda self, sym: {
            "underlying": "TSLA",
            "expiration": datetime.now(),  # expires today
            "option_type": "C",
            "strike": 650.0,
            "spot_price": 640.0,
            "historical_volatility": 25.0,
            "dividend_info": "Not Implemented",
        },
    )

    with pytest.raises(SystemExit) as excinfo:
        create_full_market_context("O:TSLA230123C00650000")
    assert excinfo.value.code == 1


def test_init_european_option_stock_market_env():
    """
    Test init_european_option_stock_market_env with direct parameters.
    """
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
    env, stock, option = init_european_option_stock_market_env(params)
    assert isinstance(env, MarketEnvironment)
    assert isinstance(stock, Stock)
    assert isinstance(option, EuropeanOption)
    assert math.isclose(env.rate, 0.05, rel_tol=1e-5)
    assert math.isclose(stock.spot, 100.0, rel_tol=1e-5)
    assert math.isclose(option.strike, 100.0, rel_tol=1e-5)


def test_init_european_option_stock_market_env_missing_keys():
    """
    Test that a KeyError is raised if required keys are missing.
    """
    params = {
        "spot": 100.0,
        # missing volatility, etc.
    }
    with pytest.raises(KeyError):
        init_european_option_stock_market_env(params)


def test_init_european_option_stock_market_env_invalid_values():
    """
    Test that a ValueError is raised if any param is invalid.
    """
    params = {
        "spot": -50.0,
        "volatility": 0.20,
        "dividend": 0.01,
        "symbol": "AAPL",
        "strike": 100.0,
        "maturity": 1.0,
        "is_call": True,
        "rate": 0.05,
    }
    with pytest.raises(ValueError):
        init_european_option_stock_market_env(params)
