#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
demo_stock.py
================
Demonstration of creating and using the Stock class.

This demo illustrates how to create a Stock instance, update its parameters
(such as spot price, volatility, and dividend yield), and handle invalid inputs.
"""

from src.underlyings.stock import Stock
from demos._utils._print_utils import (
    print_title,
    print_subtitle,
    print_info,
    print_blank_line,
)


def demo_stock() -> None:
    """
    Demonstrate usage of the Stock class.

    This function creates a Stock instance with initial parameters, then
    updates the spot price, volatility, and dividend yield. It also tests
    error handling by attempting to set an invalid spot price.
    """
    print_title("Stock Example")

    print_subtitle("Creating Stock with Spot=100, Vol=0.20, Div=0.01, Symbol='AAPL'")
    aapl = Stock(spot=100.0, volatility=0.20, dividend=0.01, symbol="AAPL")
    print_info(f"Initial stock:\n{aapl}")

    print_subtitle("Updating the Spot Price to 105.0")
    aapl.spot = 105.0
    print_info(f"After updating spot:\n{aapl}")

    print_subtitle("Updating the Volatility to 25%")
    aapl.volatility = 0.25
    print_info(f"After updating volatility:\n{aapl}")

    print_subtitle("Updating the Dividend Yield to 1.5%")
    aapl.dividend = 0.015
    print_info(f"After updating dividend:\n{aapl}")

    print_subtitle("Testing Invalid Spot (Should Raise Exception)")
    try:
        aapl.spot = -10.0
    except ValueError as e:
        print_info(f"Caught expected exception: {e}")

    print_blank_line()


if __name__ == "__main__":
    demo_stock()
