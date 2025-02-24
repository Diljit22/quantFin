#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
demo_market_environment.py
===========================
Demonstration of using the MarketEnvironment class.

This demo shows how to create a MarketEnvironment instance, update its
risk-free rate, and display the results using standardized print utilities.
"""

from src.market import MarketEnvironment
from demos._utils._print_utils import (
    print_title,
    print_subtitle,
    print_info,
    print_blank_line,
)


def demo_market_environment() -> None:
    """
    Demonstrate usage of the MarketEnvironment class.

    The demo creates a default market environment, updates the interest rate,
    and prints the results.
    """
    print_title("Market Environment Example")

    print_subtitle("Creating Default Environment")
    env = MarketEnvironment()
    print_info(f"Initial environment: {env}")

    print_subtitle("Updating the Interest Rate to 5%")
    env.rate = 0.05
    print_info(f"After setting rate to 5%: {env}")

    print_subtitle("Setting a Negative Rate (-1%)")
    env.rate = -0.01
    print_info(f"After setting rate to -1%: {env}")

    print_blank_line()


if __name__ == "__main__":
    demo_market_environment()
