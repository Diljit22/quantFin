# test_stock.py

import threading
import pytest
from src.underlyings.stock import Stock


def test_valid_stock_creation():
    """Test that a Stock instance is created with valid inputs."""
    stock = Stock(spot=100.0, volatility=0.2, dividend=0.05, symbol="AAPL")
    assert stock.spot == 100.0
    assert stock.volatility == 0.2
    assert stock.dividend == 0.05
    assert stock.symbol == "AAPL"


def test_default_symbol():
    """Test that the default symbol is 'N/A' when not provided."""
    stock = Stock(spot=100.0, volatility=0.2, dividend=0.05)
    assert stock.symbol == "N/A"


@pytest.mark.parametrize("spot", [0, -1, -100, float("inf"), float("nan")])
def test_invalid_spot(spot):
    """Test that invalid spot values raise a ValueError."""
    with pytest.raises(ValueError):
        Stock(spot=spot, volatility=0.2, dividend=0.05)


@pytest.mark.parametrize("volatility", [-0.1, -1, float("inf"), float("nan")])
def test_invalid_volatility(volatility):
    """Test that invalid volatility values raise a ValueError."""
    with pytest.raises(ValueError):
        Stock(spot=100.0, volatility=volatility, dividend=0.05)


@pytest.mark.parametrize("dividend", [-0.1, -1, float("inf"), float("nan")])
def test_invalid_dividend(dividend):
    """Test that invalid dividend values raise a ValueError."""
    with pytest.raises(ValueError):
        Stock(spot=100.0, volatility=0.2, dividend=dividend)


def test_setters():
    """Test that the setters correctly update the stock values."""
    stock = Stock(spot=100.0, volatility=0.2, dividend=0.05, symbol="AAPL")
    # Update spot, volatility, and dividend to new valid values.
    stock.spot = 120.0
    stock.volatility = 0.25
    stock.dividend = 0.03

    assert stock.spot == 120.0
    assert stock.volatility == 0.25
    assert stock.dividend == 0.03


def test_setters_invalid():
    """Test that invalid updates using setters raise a ValueError."""
    stock = Stock(spot=100.0, volatility=0.2, dividend=0.05, symbol="AAPL")
    with pytest.raises(ValueError):
        stock.spot = 0.0
    with pytest.raises(ValueError):
        stock.volatility = -0.1
    with pytest.raises(ValueError):
        stock.dividend = -0.01


def test_repr():
    """Test that __repr__ returns a string with the expected information."""
    stock = Stock(spot=100.0, volatility=0.2, dividend=0.05, symbol="AAPL")
    rep = repr(stock)
    assert "AAPL" in rep
    # Checks upto 4 decimal places.
    assert "100.0000" in rep
    assert "0.2000" in rep
    assert "0.0500" in rep


def test_thread_safety():
    """Test that concurrent updates to the stock are thread-safe."""
    stock = Stock(spot=100.0, volatility=0.2, dividend=0.05, symbol="AAPL")

    def update_stock():
        for _ in range(1000):
            # Note: the setter already validates, so we know new values are valid.
            stock.spot = stock.spot + 1.0

    threads = [threading.Thread(target=update_stock) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # The expected final spot: starting at 100.0, each of 10 threads adds 1.0 1000 times.
    expected_spot = 100.0 + 10 * 1000 * 1.0
    assert stock.spot == expected_spot
