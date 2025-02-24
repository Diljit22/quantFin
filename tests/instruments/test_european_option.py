#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_european_option.py
=======================
Pytests for EuropeanOption.

This module tests:
    - Payoff calculations for EuropeanOption.
    - Variation methods (with_strike, with_maturity, companion_option).
    - __hashable_state__ consistency.
"""

import pytest
from src.instruments import EuropeanOption


def test_european_option_payoff():
    """
    Test payoff calculation for EuropeanOption.

    Validates intrinsic payoff for both call and put options.
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

    Validates with_strike, with_maturity, and companion_option functionality.
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


def test_european_option_hashable_state():
    """
    Test that __hashable_state__ for EuropeanOption returns the expected tuple.
    """
    option = EuropeanOption(strike=100, maturity=1, is_call=True)
    state = option.__hashable_state__()
    expected_state = (100, 1, True)
    assert state == expected_state
