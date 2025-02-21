# test_market_enviroment.py

import threading
import pytest
from src.market.market_environment import MarketEnvironment


def test_default_rate():
    """Test that a MarketEnvironment created with no rate defaults to 0.0."""
    me = MarketEnvironment()
    assert me.rate == 0.0


def test_rate_setter_getter():
    """Test that setting and getting the rate works correctly."""
    me = MarketEnvironment(rate=0.05)
    assert me.rate == 0.05
    me.rate = -0.01
    assert me.rate == -0.01


def test_repr():
    """Test the string representation includes the formatted rate."""
    rate_value = 0.123456
    me = MarketEnvironment(rate=rate_value)
    rep = repr(me)
    assert "MarketEnvironment" in rep

    formatted_rate = f"{rate_value:.6f}"
    assert formatted_rate in rep


def test_thread_safety():
    """
    Test that concurrent updates to the rate are thread-safe.
    This test first updates the rate to a fixed value using multiple threads,
    then performs concurrent increments.
    """
    me = MarketEnvironment(rate=0.0)

    def set_rate(new_rate):
        me.rate = new_rate

    threads = []

    for i in range(10):
        t = threading.Thread(target=set_rate, args=(i * 0.01,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    assert me.rate in [i * 0.01 for i in range(10)]

    me.rate = 0.0

    def add_rate():
        for _ in range(1000):
            current = me.rate
            me.rate = current + 0.001

    threads = [threading.Thread(target=add_rate) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert me.rate == pytest.approx(0.0 + 10 * 1000 * 0.001, rel=1e-3)
