"""
test_parity_functions.py
========================

This file contains a suite of pytest tests for the utility functions
in parity_bounds.py and parity_implied_rate.py. It verifies the correctness
of:

    - put_call_parity: Computes the complementary option price.
    - put_call_bound: Computes option price bounds.
    - lower_bound_rate: Computes a lower bound on the risk-free rate.
    - implied_rate: Computes the implied risk-free rate via root finding.

Run these tests with:
    pytest test_parity_functions.py
"""

import math
import pytest

from src.financial_calculations.parity_bounds import (
    put_call_parity,
    put_call_bound,
    lower_bound_rate,
)
from src.financial_calculations.parity_implied_rate import implied_rate


def test_put_call_parity_complementary_price():
    """
    Test the put_call_parity function in both directions:
      - When given the put price, compute the call price.
      - When given the call price, compute the put price.
    """
    # Parameters based on the documented example.
    S = 100
    K = 110
    r = 0.08
    T = 0.5
    q = 0.01
    # Given put price.
    put_price = 6.71
    # Expected call price computed from put_call_parity (when price_call=False).
    expected_call_price = 0.5244096125126907

    # Compute call price from put price.
    computed_call = put_call_parity(put_price, S, K, r, T, q=q, price_call=False)
    assert math.isclose(
        computed_call, expected_call_price, rel_tol=1e-6
    ), f"Expected call price {expected_call_price}, got {computed_call}"

    # Now, using the computed call price, get back the put price.
    # When price_call=True, the function returns the put price.
    computed_put = put_call_parity(
        expected_call_price, S, K, r, T, q=q, price_call=True
    )
    assert math.isclose(
        computed_put, put_price, rel_tol=1e-6
    ), f"Expected put price {put_price}, got {computed_put}"


def test_put_call_bound_put():
    """
    Test the put_call_bound function for a put option.
    The lower bound is max(0, discounted_K - discounted_S) and the upper bound is discounted_K.
    """
    opPr = 2.03
    S = 36.0
    K = 37.0
    r = 0.055
    T = 0.5
    bound_call = False  # Compute bounds for a put.
    expected_lower_bound = 2.026363254477799
    expected_upper_bound = 3.03

    lower_bound, upper_bound = put_call_bound(opPr, S, K, r, T, bound_call=bound_call)
    assert math.isclose(
        lower_bound, expected_lower_bound, rel_tol=1e-6
    ), f"Expected lower bound {expected_lower_bound}, got {lower_bound}"
    assert math.isclose(
        upper_bound, expected_upper_bound, rel_tol=1e-6
    ), f"Expected upper bound {expected_upper_bound}, got {upper_bound}"


def test_lower_bound_rate_valid():
    """
    Test lower_bound_rate with valid input parameters.
    """
    call_price = 0.5287
    put_price = 6.7143
    S = 100
    K = 110
    T = 0.5
    expected_rate = 0.07058371879701723

    computed_rate = lower_bound_rate(call_price, put_price, S, K, T)
    assert math.isclose(
        computed_rate, expected_rate, rel_tol=1e-6
    ), f"Expected lower bound rate {expected_rate}, got {computed_rate}"


def test_lower_bound_rate_invalid():
    """
    Test that lower_bound_rate raises a ValueError when
    S - call_price + put_price is not positive.
    """
    call_price = 100  # Extreme call price
    put_price = 0
    S = 50
    K = 100
    T = 1

    with pytest.raises(ValueError):
        lower_bound_rate(call_price, put_price, S, K, T)


def test_implied_rate():
    """
    Test the implied_rate function with a known example.
    """
    call_price = 0.5287
    put_price = 6.7143
    S = 100
    K = 110
    T = 0.5
    q = 0.01
    expected_rate = 0.07999981808260372

    computed_rate = implied_rate(call_price, put_price, S, K, T, q=q)
    assert math.isclose(
        computed_rate, expected_rate, rel_tol=1e-6
    ), f"Expected implied rate {expected_rate}, got {computed_rate}"


def test_implied_rate_with_none_dividend():
    """
    Test implied_rate when q is provided as None.
    The function should default q to 0.0.
    For this test, we set up parameters so that the equation
      S * exp(-q*T) - K * exp(-r*T) = (call_price - put_price)
    simplifies to solving for r given q=0.0.
    """
    call_price = 10
    put_price = 5
    S = 100
    K = 100
    T = 1

    # With q=0.0, the parity equation becomes:
    #    100 - 100 * exp(-r) = (call_price - put_price) = 5,
    # so exp(-r) = 95/100, and thus r = -ln(95/100)
    expected_rate = -math.log(95 / 100)

    computed_rate = implied_rate(call_price, put_price, S, K, T, q=None)
    assert math.isclose(
        computed_rate, expected_rate, rel_tol=1e-6
    ), f"Expected implied rate {expected_rate}, got {computed_rate}"
