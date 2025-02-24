#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_stock.py
=============
Pytests for the Stock class.

This module tests the functionality of the Stock class, including:
  - Creation with valid inputs.
  - Default symbol behavior.
  - That invalid spot, volatility, and dividend values raise ValueError.
  - Setter methods for updating stock attributes.
  - String representation (__repr__) output.
  - __hashable_state__ correctness.
  - Thread safety during concurrent updates.
"""

import threading
import pytest
from src.underlyings import Stock
import math


def test_valid_stock_creation():
    """
    Test that a Stock instance is created with valid inputs.
    """
    stock = Stock(spot=100.0, volatility=0.2, dividend=0.05, symbol="AAPL")
    assert stock.spot == 100.0
    assert stock.volatility == 0.2
    assert stock.dividend == 0.05
    assert stock.symbol == "AAPL"


def test_default_symbol():
    """
    Test that the default symbol is 'N/A' when not provided.
    """
    stock = Stock(spot=100.0, volatility=0.2, dividend=0.05)
    assert stock.symbol == "N/A"


@pytest.mark.parametrize("spot", [0, -1, -100, float("inf"), float("nan")])
def test_invalid_spot(spot):
    """
    Test that invalid spot values raise a ValueError.

    Parameters
    ----------
    spot : float
        The spot price value to test.
    """
    with pytest.raises(ValueError):
        Stock(spot=spot, volatility=0.2, dividend=0.05)


@pytest.mark.parametrize("volatility", [-0.1, -1, float("inf"), float("nan")])
def test_invalid_volatility(volatility):
    """
    Test that invalid volatility values raise a ValueError.

    Parameters
    ----------
    volatility : float
        The volatility value to test.
    """
    with pytest.raises(ValueError):
        Stock(spot=100.0, volatility=volatility, dividend=0.05)


@pytest.mark.parametrize("dividend", [-0.1, -1, float("inf"), float("nan")])
def test_invalid_dividend(dividend):
    """
    Test that invalid dividend values raise a ValueError.

    Parameters
    ----------
    dividend : float
        The dividend yield value to test.
    """
    with pytest.raises(ValueError):
        Stock(spot=100.0, volatility=0.2, dividend=dividend)


def test_setters():
    """
    Test that the setter methods correctly update the stock's attributes.
    """
    stock = Stock(spot=100.0, volatility=0.2, dividend=0.05, symbol="AAPL")
    # Update spot, volatility, and dividend to new valid values.
    stock.spot = 120.0
    stock.volatility = 0.25
    stock.dividend = 0.03

    assert stock.spot == 120.0
    assert stock.volatility == 0.25
    assert stock.dividend == 0.03


def test_setters_invalid():
    """
    Test that invalid updates via setters raise a ValueError.
    """
    stock = Stock(spot=100.0, volatility=0.2, dividend=0.05, symbol="AAPL")
    with pytest.raises(ValueError):
        stock.spot = 0.0
    with pytest.raises(ValueError):
        stock.volatility = -0.1
    with pytest.raises(ValueError):
        stock.dividend = -0.01


def test_repr():
    """
    Test that __repr__ returns a string with the expected information.
    """
    stock = Stock(spot=100.0, volatility=0.2, dividend=0.05, symbol="AAPL")
    rep = repr(stock)
    # Check that the representation contains the symbol and correctly formatted numbers.
    assert "AAPL" in rep
    # Verify that spot, volatility, and dividend are formatted to four decimal places.
    assert "100.0000" in rep
    assert "0.2000" in rep
    assert "0.0500" in rep


def test_hashable_state():
    """
    Test that __hashable_state__ returns a tuple with the expected stock parameters.
    """
    stock = Stock(spot=100.0, volatility=0.2, dividend=0.05, symbol="AAPL")
    state = stock.__hashable_state__()
    expected_state = (100.0, 0.2, 0.05)
    assert state == expected_state


def test_thread_safety():
    """
    Test that concurrent updates to the stock are thread-safe.

    This test concurrently updates the stock's spot price using multiple threads
    and verifies that the final spot price is as expected.
    """
    stock = Stock(spot=100.0, volatility=0.2, dividend=0.05, symbol="AAPL")

    def update_stock():
        for _ in range(1000):
            # Increment the spot price by 1.0.
            stock.spot = stock.spot + 1.0

    threads = [threading.Thread(target=update_stock) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    expected_spot = 100.0 + 10 * 1000 * 1.0
    assert stock.spot == expected_spot
