#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_market_environment.py
==========================
Pytests for the MarketEnvironment class.

This module tests the functionality of the MarketEnvironment class in the market
package, including:
  - Default initialization.
  - Getter and setter for the risk-free rate.
  - String representation.
  - Thread-safety during concurrent updates.
"""

import threading
import pytest
from src.market import MarketEnvironment


def test_default_rate():
    """
    Test that a MarketEnvironment created without specifying a rate defaults to 0.0.
    """
    me = MarketEnvironment()
    assert me.rate == 0.0


def test_rate_setter_getter():
    """
    Test that setting and retrieving the risk-free rate works correctly.
    """
    me = MarketEnvironment(rate=0.05)
    # Verify initial rate
    assert me.rate == 0.05

    # Update the rate and verify
    me.rate = -0.01
    assert me.rate == -0.01


def test_repr():
    """
    Test that the string representation includes the class name and
    correctly formatted rate.
    """
    rate_value = 0.123456
    me = MarketEnvironment(rate=rate_value)
    rep = repr(me)

    # Check that the representation includes the class name.
    assert "MarketEnvironment" in rep

    # Verify that the rate is formatted to six decimal places.
    formatted_rate = f"{rate_value:.6f}"
    assert formatted_rate in rep


def test_thread_safety():
    """
    Test that concurrent updates to the risk-free rate are thread-safe.

    The test first sets the rate concurrently to different fixed values and
    verifies that the final rate is one of the expected values.
    Then it resets the rate and performs concurrent increments,
    verifying that the final rate is approximately the expected cumulative sum.

    """
    me = MarketEnvironment(rate=0.0)

    def set_rate(new_rate):
        me.rate = new_rate

    # Concurrently set the rate using 10 threads.
    threads = []
    for i in range(10):
        t = threading.Thread(target=set_rate, args=(i * 0.01,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    # Expected values are 0.0, 0.01, ..., 0.09; the final rate should be one of them.
    expected_values = [i * 0.01 for i in range(10)]
    assert me.rate in expected_values

    # Reset the rate for the next part of the test.
    me.rate = 0.0

    def add_rate():
        # Increment the rate 1000 times by 0.001 each time.
        for _ in range(1000):
            current = me.rate
            me.rate = current + 0.001

    # Run 10 threads concurrently to increment the rate.
    threads = [threading.Thread(target=add_rate) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Expected final rate: 10 threads * 1000 increments * 0.001 per increment.
    expected_total = 10 * 1000 * 0.001
    assert me.rate == pytest.approx(expected_total, rel=1e-3)
