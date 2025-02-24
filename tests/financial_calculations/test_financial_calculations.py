#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_parity_functions.py
========================
Pytests for the put-call functions in the financial_calculations package.


This module tests the following functions:
  - put_call_parity: Computes the complementary option price via put-call parity.
  - put_call_bound: Computes naive lower/upper bounds for options.
  - lower_bound_rate: Bound the risk-free rate below via put-call inequality.
  - implied_rate: Numerically solves for the implied risk-free rate via Brent's method.
  - perpetual_put: Prices a perpetual put option using a closed-form solution.
"""

import math
import pytest

from src.financial_calculations import (
    put_call_parity,
    put_call_bound,
    lower_bound_rate,
    implied_rate,
    perpetual_put,
)


def test_put_call_parity_put_to_call() -> None:
    """
    Test put_call_parity for: put price -> call price.

    The put-call parity formula is:
        C - P = S * exp(-q*T) - K * exp(-r*T)
    Given a put price, the call price is computed as:
        call_price = put_price + (S * exp(-q*T) - K * exp(-r*T))
    """
    put_price = 6.71
    computed_call = put_call_parity(
        option_price=put_price, S=100, K=110, r=0.08, T=0.5, q=0.01, price_call=False
    )
    expected_call = 0.5244096125126907
    assert computed_call == pytest.approx(expected_call, rel=1e-3)


def test_put_call_parity_call_to_put() -> None:
    """
    Test put_call_parity for: call price -> put price.

    Given a call price, the put price is computed as:
        put_price = call_price - (S * exp(-q*T) - K * exp(-r*T))
    """
    call_price = 0.5244096125126907
    computed_put = put_call_parity(
        option_price=call_price, S=100, K=110, r=0.08, T=0.5, q=0.01, price_call=True
    )
    expected_put = 6.71
    assert computed_put == pytest.approx(expected_put, rel=1e-3)


@pytest.mark.parametrize(
    "option_price, S, K, r, T, bound_call, expected",
    [
        (2.03, 36.0, 37.0, 0.055, 0.5, False, (2.026363254477799, 3.03)),
    ],
)
def test_put_call_bound(
    option_price: float,
    S: float,
    K: float,
    r: float,
    T: float,
    bound_call: bool,
    expected: tuple,
) -> None:
    """
    Test put_call_bound for computing lower and upper bounds for an option.

    Parameters
    ----------
    option_price : float
        Known price of contract.
    S : float
        Underlying spot price.
    K : float
        Strike price.
    r : float
        Continuously compounded risk-free rate.
    T : float
        Time to maturity in years.
    bound_call : bool
        If True, compute bounds for a call; otherwise, for a put.
    expected : tuple of float
        Expected (lower_bound, upper_bound).

    """
    bounds = put_call_bound(option_price, S, K, r, T, bound_call=bound_call)
    lower_expected, upper_expected = expected
    assert bounds[0] == pytest.approx(lower_expected, rel=1e-6)
    assert bounds[1] == pytest.approx(upper_expected, rel=1e-6)


def test_lower_bound_rate_valid() -> None:
    """
    Test that lower_bound_rate computes a valid lower bound on the risk-free rate.

    Uses the example:
        call_price = 0.5287, put_price = 6.7143, S = 100, K = 110, T = 0.5,
    which should yield a lower bound approximately equal to 0.07058371879701723.
    """
    computed_rate = lower_bound_rate(
        call_price=0.5287, put_price=6.7143, S=100, K=110, T=0.5
    )
    expected_rate = 0.07058371879701723
    assert computed_rate == pytest.approx(expected_rate, rel=1e-3)


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


def test_implied_rate_valid() -> None:
    """
    Test implied_rate with valid inputs.
    """
    computed_rate = implied_rate(
        call_price=0.5287, put_price=6.7143, S=100, K=110, T=0.5, q=0.01
    )
    expected_rate = 0.07999981808260372
    assert computed_rate == pytest.approx(expected_rate, rel=1e-3)


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


def test_perpetual_put() -> None:
    """
    Test perpetual_put with example parameters.
    """
    computed_value = perpetual_put(S=150, K=100, r=0.08, vol=0.2, q=0.005)
    expected_value = 1.8344292693352158
    assert computed_value == pytest.approx(expected_value, rel=1e-3)
