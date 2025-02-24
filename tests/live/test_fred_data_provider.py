#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_fred_data_provider.py
==========================
Pytest suite for the FredDataProvider class.

This module tests key methods of FredDataProvider, including:
  - get_short_term_rate
  - get_generic_rate
  - get_long_term_rate (via interpolation of a dummy yield curve)
"""

import math
import pytest
import requests
from src.live import FredDataProvider


# Dummy response to simulate requests.get responses.
class DummyResponse:
    def __init__(self, json_data, status_code=200, text="dummy text"):
        self._json_data = json_data
        self.status_code = status_code
        self.text = text

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code != 200:
            raise requests.RequestException("Error")


def dummy_get(url, params, timeout):
    """
    Dummy requests.get replacement that returns fixed observation data.
    """
    # For simplicity, always return an observation with value "2.5"
    dummy_data = {"observations": [{"date": "2023-01-01", "value": "2.5"}]}
    return DummyResponse(dummy_data, status_code=200)


def test_get_short_term_rate(monkeypatch):
    """
    Test get_short_term_rate returns a valid rate from dummy data.
    """
    monkeypatch.setattr(requests, "get", dummy_get)
    provider = FredDataProvider(fred_api_key="dummy")
    rate = provider.get_short_term_rate("DTB3")
    # 2.5% should become 0.025 as a decimal.
    assert math.isclose(rate, 0.025, rel_tol=1e-3)


def test_get_generic_rate(monkeypatch):
    """
    Test get_generic_rate returns a valid rate from dummy data.
    """
    monkeypatch.setattr(requests, "get", dummy_get)
    provider = FredDataProvider(fred_api_key="dummy")
    rate = provider.get_generic_rate("DTB3")
    assert math.isclose(rate, 0.025, rel_tol=1e-3)


def dummy_yield_curve(series_map):
    """
    Return a dummy yield curve: maturities mapped to yields in decimal.
    """
    return {1: 0.01, 2: 0.02, 3: 0.03, 5: 0.05, 7: 0.06, 10: 0.07, 20: 0.08, 30: 0.09}


def test_get_long_term_rate(monkeypatch):
    """
    Test get_long_term_rate by patching get_yield_curve to return a dummy yield curve.
    """
    monkeypatch.setattr(
        FredDataProvider, "get_yield_curve", lambda self, sm: dummy_yield_curve(sm)
    )
    provider = FredDataProvider(fred_api_key="dummy")
    # For maturity 5, our dummy yield curve returns 0.05 exactly.
    long_rate = provider.get_long_term_rate(5.0)
    assert math.isclose(long_rate, 0.05, rel_tol=1e-3)
