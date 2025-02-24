#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_option_evaluation.py
=========================
Pytests for the OptionEvaluation class.

This module tests:
  - Creation and attribute access.
  - Default values for optional attributes.
  - String representation (__str__) formatting.
  - Immutability of the OptionEvaluation instance.
"""

import math
import pytest
from dataclasses import FrozenInstanceError
from src.results import OptionEvaluation


def test_option_evaluation_creation() -> None:
    """
    Test creation of an OptionEvaluation instance with all parameters.
    """
    eval_instance = OptionEvaluation(
        model="BlackScholesMerton",
        technique="ClosedForm",
        price=10.1234,
        delta=0.123,
        gamma=0.456,
        vega=0.789,
        theta=-0.123,
        rho=0.456,
        implied_vol=0.789,
        instrument_data={"strike": 100, "maturity": 1.0, "is_call": True},
        underlying_data={"spot": 100, "dividend": 0.02, "volatility": 0.2},
    )
    # Verify that the instance attributes are correctly set.
    assert eval_instance.model == "BlackScholesMerton"
    assert eval_instance.technique == "ClosedForm"
    assert math.isclose(eval_instance.price, 10.1234, rel_tol=1e-4)
    assert math.isclose(eval_instance.delta, 0.123, rel_tol=1e-4)
    assert math.isclose(eval_instance.gamma, 0.456, rel_tol=1e-4)
    assert math.isclose(eval_instance.vega, 0.789, rel_tol=1e-4)
    assert math.isclose(eval_instance.theta, -0.123, rel_tol=1e-4)
    assert math.isclose(eval_instance.rho, 0.456, rel_tol=1e-4)
    assert math.isclose(eval_instance.implied_vol, 0.789, rel_tol=1e-4)
    assert eval_instance.instrument_data == {
        "strike": 100,
        "maturity": 1.0,
        "is_call": True,
    }
    assert eval_instance.underlying_data == {
        "spot": 100,
        "dividend": 0.02,
        "volatility": 0.2,
    }


def test_option_evaluation_defaults() -> None:
    """
    Test that optional attributes default to None when not provided.
    """
    eval_instance = OptionEvaluation(model="ModelX", technique="Test", price=5.0)
    assert eval_instance.delta is None
    assert eval_instance.gamma is None
    assert eval_instance.vega is None
    assert eval_instance.theta is None
    assert eval_instance.rho is None
    assert eval_instance.implied_vol is None
    assert eval_instance.instrument_data is None
    assert eval_instance.underlying_data is None


def test_option_evaluation_str() -> None:
    """
    Test that the string representation (__str__) contains the expected key information.
    """
    eval_instance = OptionEvaluation(
        model="TestModel",
        technique="TestTech",
        price=15.0,
        delta=0.1,
        gamma=0.2,
        vega=0.3,
        theta=-0.05,
        rho=0.04,
        implied_vol=0.25,
        instrument_data="TestInstrument",
        underlying_data="TestUnderlying",
    )
    rep = str(eval_instance)
    # Verify that the key strings appear in the output.
    assert "TestModel" in rep
    assert "TestTech" in rep
    assert "15.0000" in rep
    assert "0.1000" in rep
    assert "0.2000" in rep
    assert "0.3000" in rep
    assert "-0.0500" in rep
    assert "0.0400" in rep
    assert "0.2500" in rep
    assert "TestInstrument" in rep
    assert "TestUnderlying" in rep


def test_option_evaluation_immutability() -> None:
    """
    Test that the OptionEvaluation instance is immutable.

    Attempts to modify an attribute should raise a FrozenInstanceError.
    """
    eval_instance = OptionEvaluation(model="Test", technique="Test", price=5.0)
    with pytest.raises(FrozenInstanceError):
        eval_instance.price = 10.0
