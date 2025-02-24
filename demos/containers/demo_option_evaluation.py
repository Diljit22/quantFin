#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
demo_option_evaluation.py
=========================
Demonstration of using the OptionEvaluation class.

This demo illustrates how to create an OptionEvaluation instance with
various evaluation parameters and displays the results.
"""

from src.results.option_evaluation import OptionEvaluation
from demos._utils._print_utils import (
    print_title,
    print_subtitle,
    print_info,
    print_blank_line,
)


def demo_option_evaluation() -> None:
    """
    Demonstrate usage of the OptionEvaluation class.
    """
    print_title("Option Evaluation Demo")

    # Create an OptionEvaluation instance with full details.
    evaluation = OptionEvaluation(
        model="BlackScholesMerton",
        technique="ClosedForm",
        price=12.3456,
        delta=0.1234,
        gamma=0.2345,
        vega=0.3456,
        theta=-0.0456,
        rho=0.0567,
        implied_vol=0.2345,
        instrument_data={"strike": 100, "maturity": 1.0, "is_call": True},
        underlying_data={"spot": 100, "dividend": 0.02, "volatility": 0.2},
    )

    print_subtitle("Created Option Evaluation Instance")
    print_info(str(evaluation))
    print_blank_line()

    # Demonstrate accessing individual attributes.
    print_subtitle("Accessing Individual Attributes")
    print_info(f"Model: {evaluation.model}")
    print_info(f"Technique: {evaluation.technique}")
    print_info(f"Price: {evaluation.price:.4f}")
    print_info(f"Delta: {evaluation.delta:.4f}")
    print_info(f"Gamma: {evaluation.gamma:.4f}")
    print_info(f"Vega: {evaluation.vega:.4f}")
    print_info(f"Theta: {evaluation.theta:.4f}")
    print_info(f"Rho: {evaluation.rho:.4f}")
    print_info(f"Implied Volatility: {evaluation.implied_vol:.4f}")
    print_info(f"Instrument Data: {evaluation.instrument_data}")
    print_info(f"Underlying Data: {evaluation.underlying_data}")
    print_blank_line()


if __name__ == "__main__":
    demo_option_evaluation()
