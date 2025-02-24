#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_market_rate_provider.py
============================
Pytest suite for the market_rate_provider module.

This module tests the function that retrieves a long-term rate via FredDataProvider
and returns it as a MarketEnvironment instance.
"""

import math
import pytest
from src.live import get_market_environment_by_maturity
from src.market import MarketEnvironment


def dummy_get_long_term_rate(self, maturity: float) -> float:
    """
    Dummy method to simulate returning a long-term rate for a given maturity.
    For testing, simply return maturity/100.
    """
    return maturity / 100.0


def test_get_market_environment_by_maturity(monkeypatch):
    """
    Test that get_market_environment_by_maturity returns a MarketEnvironment instance
    with the correct rate.
    """
    # Patch FredDataProvider.get_long_term_rate to use the dummy implementation.
    from src.live import FredDataProvider

    monkeypatch.setattr(
        FredDataProvider, "get_long_term_rate", dummy_get_long_term_rate
    )

    market_env = get_market_environment_by_maturity(5.0)
    # Expected rate: 5.0 / 100 = 0.05.
    assert isinstance(market_env, MarketEnvironment)
    assert math.isclose(market_env.rate, 0.05, rel_tol=1e-3)
