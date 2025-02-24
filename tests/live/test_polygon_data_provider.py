#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_polygon_data_provider.py
=============================
Pytest suite for the PolygonDataProvider class.

This module tests key methods of PolygonDataProvider, including:
  - get_spot_price (with fallback to previous close)
  - parse_option_symbol
  - get_option_data
"""

import math
import pytest
import requests
from datetime import datetime
from src.live import PolygonDataProvider


# Dummy response class to simulate requests.get() output.
class DummyResponse:
    def __init__(self, json_data, status_code=200, text="dummy text"):
        self._json_data = json_data
        self.status_code = status_code
        self.text = text  # needed for debug logging

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code != 200:
            raise requests.RequestException(f"HTTP {self.status_code} Error")


def dummy_spot_price(url, params, timeout):
    """
    Return a fake realtime spot price from the 'last' field.
    """
    dummy_data = {"last": {"price": 123.45}}
    return DummyResponse(dummy_data, 200)


def dummy_previous_close(url, params, timeout):
    """
    Return a fake previous close price from the 'results' field.
    """
    dummy_data = {"results": [{"c": 120.0}]}
    return DummyResponse(dummy_data, 200)


def dummy_option_data(url, params, timeout):
    """
    Return a fake option data for historical prices or fallback logic.
    """
    # We assume either historical or last trade endpoint; for now, we simulate historical prices
    # with a 'results' key in the aggregator style:
    dummy_data = {
        "results": [
            {"c": 100.0},
            {"c": 101.0},
            {"c": 99.0},
            {"c": 102.0},
        ]
    }
    return DummyResponse(dummy_data, 200)


def dummy_forbidden(url, params, timeout):
    """
    Return a 403 to simulate forbidden realtime data.
    """
    return DummyResponse({}, 403)


def test_get_spot_price_realtime(monkeypatch):
    """
    Test that get_spot_price fetches a realtime price if not forbidden.
    """
    monkeypatch.setattr(requests, "get", dummy_spot_price)
    provider = PolygonDataProvider(polygon_api_key="dummy")
    price = provider.get_spot_price("AAPL")
    assert math.isclose(price, 123.45, rel_tol=1e-3)


def test_get_spot_price_fallback(monkeypatch):
    """
    Test that get_spot_price falls back to get_previous_close if a 403 is encountered.
    """

    # First call returns 403, second call returns previous close
    def dummy_requests_get(url, params, timeout):
        # If first call, 403; if second call, return normal previous close
        if "last/trade" in url:
            return dummy_forbidden(url, params, timeout)
        else:
            return dummy_previous_close(url, params, timeout)

    monkeypatch.setattr(requests, "get", dummy_requests_get)
    provider = PolygonDataProvider(polygon_api_key="dummy")
    price = provider.get_spot_price("AAPL")
    # Should fallback to the prevClose of 120.0
    assert math.isclose(price, 120.0, rel_tol=1e-3)


def test_parse_option_symbol():
    """
    Test parse_option_symbol with a well-formed OSI option symbol.
    """
    provider = PolygonDataProvider(polygon_api_key="dummy")
    result = provider.parse_option_symbol("O:SPY251219C00650000")
    # Underlying: SPY
    # Expiration: 2025-12-19
    # Option Type: C
    # Strike: 650.00
    assert result["underlying"] == "SPY"
    assert result["option_type"] == "C"
    assert math.isclose(result["strike"], 650.0, rel_tol=1e-3)
    # Check expiration year:
    exp = result["expiration"]
    assert isinstance(exp, datetime)
    assert exp.year == 2025 and exp.month == 12 and exp.day == 19


def test_get_option_data(monkeypatch):
    """
    Test get_option_data, ensuring it fetches spot price and historical data
    and computes historical volatility.
    """

    def dummy_spot(url, params, timeout):
        # Return a valid 'last' price
        data = {"last": {"price": 200.0}}
        return DummyResponse(data, 200)

    def dummy_hist(url, params, timeout):
        # Return aggregator style historical data
        data = {"results": [{"c": 100.0}, {"c": 101.0}, {"c": 99.0}, {"c": 102.0}]}
        return DummyResponse(data, 200)

    # We'll patch calls differently depending on endpoint
    def mock_requests_get(url, params, timeout):
        if "/last/trade/" in url:
            return dummy_spot(url, params, timeout)
        elif "/range/1/day/" in url:
            return dummy_hist(url, params, timeout)
        # default fallback
        return DummyResponse({}, 200)

    monkeypatch.setattr(requests, "get", mock_requests_get)
    provider = PolygonDataProvider(polygon_api_key="dummy")
    data = provider.get_option_data("O:TSLA240119C00100000")
    assert data["underlying"] == "TSLA"
    # spot_price should be 200.0
    assert math.isclose(data["spot_price"], 200.0, rel_tol=1e-3)
    # historical_volatility is computed from 100,101,99,102 => check it isn't 0
    assert data["historical_volatility"] > 0
    # strike = 100.0
    assert math.isclose(data["strike"], 100.0, rel_tol=1e-3)
    # option_type = "C"
    assert data["option_type"] == "C"


def test_main_cli_no_symbol(monkeypatch, capsys):
    """
    Test the main_cli with no symbol in stock mode should sys.exit(1).
    """
    # Patch sys.argv
    monkeypatch.setattr("sys.argv", ["polygon_data_provider.py", "--mode", "stock"])
    provider = PolygonDataProvider(polygon_api_key="dummy")

    # We expect SystemExit because no --symbol is provided
    with pytest.raises(SystemExit) as excinfo:
        provider.main_cli()
    assert excinfo.type == SystemExit
    assert excinfo.value.code == 1
