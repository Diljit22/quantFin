"""
test_options.py
===============

Pytest for EuropeanOption and EuropeanOptionVector classes.
"""

import numpy as np
import pytest
from src.instruments.european_option import EuropeanOption
from src.instruments.european_option_vector import EuropeanOptionVector


def test_european_option_payoff():
    """
    Test payoff calculation for EuropeanOption.

    Validates intrinsic payoff for call and put options.
    """
    call_option = EuropeanOption(strike=100.0, maturity=1.0, is_call=True)
    assert call_option.payoff(120.0) == 20.0
    assert call_option.payoff(90.0) == 0.0

    put_option = EuropeanOption(strike=100.0, maturity=1.0, is_call=False)
    assert put_option.payoff(80.0) == 20.0
    assert put_option.payoff(110.0) == 0.0


def test_european_option_variation_methods():
    """
    Test variation methods for EuropeanOption.

    Validates with_strike, with_maturity, and companion_option.
    """
    option = EuropeanOption(strike=100.0, maturity=1.0, is_call=True)

    new_option = option.with_strike(110.0)
    assert new_option.strike == 110.0
    assert new_option.maturity == option.maturity
    assert new_option.is_call == option.is_call

    updated_maturity = option.with_maturity(2.0)
    assert updated_maturity.maturity == 2.0
    assert updated_maturity.strike == option.strike

    companion = option.companion_option
    assert companion.is_call == (not option.is_call)
    assert companion.strike == option.strike
    assert companion.maturity == option.maturity


def test_european_option_vector_payoff():
    """
    Test payoff calculation for EuropeanOptionVector.

    Validates vectorized payoff for scalar and array spot prices.
    """
    strikes = np.array([90.0, 100.0, 110.0])
    vector_option = EuropeanOptionVector(strikes=strikes, maturity=1.0, is_call=True)

    payoff_scalar = vector_option.payoff(120.0)
    expected_scalar = np.array([[30.0, 20.0, 10.0]])
    np.testing.assert_array_equal(payoff_scalar, expected_scalar)

    spot_prices = np.array([80.0, 120.0])
    payoff_array = vector_option.payoff(spot_prices)
    expected_array = np.array([[0.0, 0.0, 0.0], [30.0, 20.0, 10.0]])
    np.testing.assert_array_equal(payoff_array, expected_array)


def test_european_option_vector_variation_methods():
    """
    Test variation methods for EuropeanOptionVector.

    Validates with_strike, with_maturity, and companion_option.
    """
    strikes = np.array([90.0, 100.0, 110.0])
    vector_option = EuropeanOptionVector(strikes=strikes, maturity=1.0, is_call=True)

    new_strikes = np.array([110.0, 115.0])
    new_vector = vector_option.with_strike(new_strikes)
    np.testing.assert_array_equal(new_vector.strikes, new_strikes)

    updated_maturity_vector = vector_option.with_maturity(1.5)
    assert updated_maturity_vector.maturity == 1.5
    np.testing.assert_array_equal(
        updated_maturity_vector.strikes, vector_option.strikes
    )

    companion_vector = vector_option.companion_option
    assert companion_vector.is_call == (not vector_option.is_call)
    np.testing.assert_array_equal(companion_vector.strikes, vector_option.strikes)
    assert companion_vector.maturity == vector_option.maturity
