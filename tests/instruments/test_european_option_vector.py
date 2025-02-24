#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_european_option_vector.py
=============================
Pytests for EuropeanOptionVector.

This module tests:
    - Vectorized payoff calculations for EuropeanOptionVector.
    - Variation methods (with_strike, with_maturity, companion_option).
    - __hashable_state__ behavior (if applicable).
"""

import numpy as np
import pytest
from src.instruments import EuropeanOptionVector


def test_european_option_vector_payoff():
    """
    Test payoff calculation for EuropeanOptionVector.

    Validates vectorized payoff for both scalar and array spot prices.
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

    Validates with_strike, with_maturity, and companion_option functionality.
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
