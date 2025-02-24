#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_base_option.py
===================
Pytests for BaseOption and common behaviors across option classes.

This module tests:
    - That BaseOption is abstract and cannot be instantiated directly.
    - Error handling for invalid parameters.
    - Common properties such as option_type, intrinsic_payoff, and __hashable_state__.
"""

import pytest
from src.instruments import BaseOption, AmericanOption, EuropeanOption


# -----------------------------------------------------------------------------
# Dummy subclass for testing purposes
# -----------------------------------------------------------------------------
class DummyOption(BaseOption):
    def payoff(self, spot_price: float) -> float:
        return self.intrinsic_payoff(spot_price)


def test_base_option_cannot_instantiate():
    """
    Test that BaseOption is abstract and cannot be instantiated directly.
    """
    with pytest.raises(TypeError):
        _ = BaseOption(strike=100, maturity=1, is_call=True)


def test_invalid_strike_and_maturity():
    """
    Test that invalid strike and maturity values raise ValueError.
    """
    from src.instruments.american_option import AmericanOption
    from src.instruments.european_option import EuropeanOption

    with pytest.raises(ValueError):
        _ = AmericanOption(strike=0, maturity=1, is_call=True)
    with pytest.raises(ValueError):
        _ = EuropeanOption(strike=100, maturity=0, is_call=False)
    with pytest.raises(ValueError):
        _ = DummyOption(strike=-50, maturity=1, is_call=True)
    with pytest.raises(ValueError):
        _ = DummyOption(strike=100, maturity=-2, is_call=False)


def test_option_type():
    """
    Test that the option_type property returns 'Call' for is_call True
    and 'Put' for is_call False.
    """
    from src.instruments.american_option import AmericanOption
    from src.instruments.european_option import EuropeanOption

    call_opt = AmericanOption(strike=100, maturity=1, is_call=True)
    put_opt = EuropeanOption(strike=100, maturity=1, is_call=False)
    assert call_opt.option_type == "Call"
    assert put_opt.option_type == "Put"


@pytest.mark.parametrize("option_class", [AmericanOption, EuropeanOption, DummyOption])
def test_intrinsic_payoff_call(option_class):
    """
    Test that intrinsic_payoff for a call option returns max(spot - strike, 0)
    and that payoff() delegates to intrinsic_payoff.
    """
    opt = option_class(strike=100, maturity=1, is_call=True)
    spot = 120
    expected_payoff = 20
    assert opt.intrinsic_payoff(spot) == expected_payoff
    assert opt.payoff(spot) == expected_payoff


@pytest.mark.parametrize("option_class", [AmericanOption, EuropeanOption, DummyOption])
def test_intrinsic_payoff_put(option_class):
    """
    Test that intrinsic_payoff for a put option returns max(strike - spot, 0)
    and that payoff() delegates to intrinsic_payoff.
    """
    opt = option_class(strike=100, maturity=1, is_call=False)
    spot = 80
    expected_payoff = 20
    assert opt.intrinsic_payoff(spot) == expected_payoff
    assert opt.payoff(spot) == expected_payoff


def test_repr_contains_key_info():
    """
    Test that the __repr__ for a concrete option contains key information.
    """
    from src.instruments.american_option import AmericanOption

    opt = AmericanOption(strike=100, maturity=1, is_call=True)
    rep = repr(opt)
    assert "AmericanOption" in rep
    assert "100" in rep
    assert "1" in rep
    assert isinstance(rep, str) and len(rep) > 0


def test_hashable_state():
    """
    Test that __hashable_state__ returns a consistent tuple representing the option.
    """
    from src.instruments.european_option import EuropeanOption

    option = EuropeanOption(strike=100, maturity=1, is_call=True)
    state = option.__hashable_state__()
    expected_state = (100, 1, True)
    assert state == expected_state
