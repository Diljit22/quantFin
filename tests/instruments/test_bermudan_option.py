#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_bermudan_option.py
=======================
Pytests for BermudanOption.

This module tests:
    - Payoff calculations for BermudanOption.
    - The exercise_dates property and can_exercise method.
    - Error handling for invalid exercise dates.
    - __hashable_state__ consistency.
"""

import numpy as np
import pytest
from src.instruments import BermudanOption


def test_bermudan_option_payoff():
    """
    Test payoff calculation for BermudanOption.

    Validates that the intrinsic payoff is computed correctly for
    both call and put options.
    """
    exercise_dates = [0.5, 1.0]
    bermudan = BermudanOption(
        strike=100.0, maturity=1.0, is_call=True, exercise_dates=exercise_dates
    )
    assert bermudan.payoff(120.0) == 20.0
    assert bermudan.payoff(90.0) == 0.0

    bermudan_put = BermudanOption(
        strike=100.0, maturity=1.0, is_call=False, exercise_dates=exercise_dates
    )
    assert bermudan_put.payoff(80.0) == 20.0
    assert bermudan_put.payoff(110.0) == 0.0


def test_bermudan_option_exercise_dates():
    """
    Test the exercise_dates property and can_exercise method for BermudanOption.

    Validates that exercise_dates are sorted and that can_exercise correctly identifies
    valid exercise times.
    """
    exercise_dates = [0.5, 1.0, 0.75]
    bermudan = BermudanOption(
        strike=100.0, maturity=1.0, is_call=True, exercise_dates=exercise_dates
    )

    expected_dates = np.array([0.5, 0.75, 1.0])
    np.testing.assert_array_equal(bermudan.exercise_dates, expected_dates)

    # Test can_exercise: time near 0.75 should return True, otherwise False.
    assert bermudan.can_exercise(0.75 + 1e-6)
    assert not bermudan.can_exercise(0.8)


def test_bermudan_option_invalid_exercise_dates():
    """
    Test that BermudanOption raises ValueError for invalid exercise dates.

    Invalid cases include an empty exercise_dates list, dates <= 0,
    and dates exceeding the maturity.
    """
    with pytest.raises(ValueError):
        _ = BermudanOption(strike=100.0, maturity=1.0, is_call=True, exercise_dates=[])

    with pytest.raises(ValueError):
        _ = BermudanOption(
            strike=100.0, maturity=1.0, is_call=True, exercise_dates=[0.0, 0.5]
        )

    with pytest.raises(ValueError):
        _ = BermudanOption(
            strike=100.0, maturity=1.0, is_call=True, exercise_dates=[0.5, 1.5]
        )


def test_bermudan_option_hashable_state():
    """
    Test that __hashable_state__ for BermudanOption returns the expected tuple.
    """
    exercise_dates = [0.5, 1.0]
    bermudan = BermudanOption(
        strike=100.0, maturity=1.0, is_call=True, exercise_dates=exercise_dates
    )
    state = bermudan.__hashable_state__()
    expected_state = (100.0, 1.0, True, tuple(sorted(exercise_dates)))
    assert state == expected_state
