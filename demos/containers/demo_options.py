#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
demo_options.py
===============
Demonstration of using the option instrument classes.

This demo illustrates how to create and manipulate:
    - EuropeanOption
    - AmericanOption
    - BermudanOption
    - EuropeanOptionVector

It shows how to compute payoffs, update parameters, and generate companion options.
"""

import numpy as np

from src.instruments import (
    AmericanOption,
    BermudanOption,
    EuropeanOption,
    EuropeanOptionVector,
)

from demos._utils._print_utils import (
    print_title,
    print_subtitle,
    print_info,
    print_blank_line,
)


def demo_european_option() -> None:
    """
    Demonstrate usage of EuropeanOption.
    """
    print_title("European Option Demo")

    print_subtitle("Creating a European Call Option")
    euro_call = EuropeanOption(strike=100.0, maturity=1.0, is_call=True)
    print_info(f"European Call Option: {euro_call}")
    print_info(f"Payoff at spot=120: {euro_call.payoff(120)}")

    print_subtitle("Updating Strike and Maturity")
    updated_euro = euro_call.with_strike(110.0).with_maturity(2.0)
    print_info(f"Updated European Option: {updated_euro}")
    print_info(f"Payoff at spot=130: {updated_euro.payoff(130)}")

    print_subtitle("Generating Companion Option")
    companion = euro_call.companion_option
    print_info(f"Companion Option: {companion}")
    print_info(f"Companion payoff at spot=90: {companion.payoff(90)}")

    print_blank_line()


def demo_american_option() -> None:
    """
    Demonstrate usage of AmericanOption.
    """
    print_title("American Option Demo")

    print_subtitle("Creating an American Put Option")
    amer_put = AmericanOption(strike=100.0, maturity=1.0, is_call=False)
    print_info(f"American Put Option: {amer_put}")
    print_info(f"Payoff at spot=80: {amer_put.payoff(80)}")

    print_subtitle("Generating Companion Option")
    companion = amer_put.companion_option
    print_info(f"Companion Option: {companion}")
    print_info(f"Companion payoff at spot=80: {companion.payoff(80)}")

    print_blank_line()


def demo_bermudan_option() -> None:
    """
    Demonstrate usage of BermudanOption.
    """
    print_title("Bermudan Option Demo")

    exercise_dates = [0.5, 1.0]
    print_subtitle("Creating a Bermudan Call Option")
    bermudan_call = BermudanOption(
        strike=100.0, maturity=1.0, is_call=True, exercise_dates=exercise_dates
    )
    print_info(f"Bermudan Call Option: {bermudan_call}")
    print_info(f"Payoff at spot=120: {bermudan_call.payoff(120)}")
    print_info(f"Exercise Dates: {bermudan_call.exercise_dates}")

    print_subtitle("Testing Exercise Capability")
    t_test = 0.5 + 1e-6
    print_info(f"Can exercise at t={t_test}: {bermudan_call.can_exercise(t_test)}")

    print_blank_line()


def demo_european_option_vector() -> None:
    """
    Demonstrate usage of EuropeanOptionVector.
    """
    print_title("European Option Vector Demo")

    print_subtitle("Creating a Vectorized European Call Option")
    strikes = np.array([90.0, 100.0, 110.0])
    euro_vector = EuropeanOptionVector(strikes=strikes, maturity=1.0, is_call=True)
    print_info(f"European Option Vector: {euro_vector}")

    print_subtitle("Calculating Vectorized Payoff")
    spot = 120
    payoff = euro_vector.payoff(spot)
    print_info(f"Payoff at spot={spot}: {payoff}")

    print_subtitle("Updating Strike Array")
    new_strikes = np.array([110.0, 115.0])
    updated_vector = euro_vector.with_strike(new_strikes)
    print_info(f"Updated Option Vector: {updated_vector}")
    print_info(f"New payoff at spot=130: {updated_vector.payoff(130)}")

    print_subtitle("Generating Companion Option Vector")
    companion_vector = euro_vector.companion_option
    print_info(f"Companion Option Vector: {companion_vector}")
    print_info(f"Companion payoff at spot=90: {companion_vector.payoff(90)}")

    print_blank_line()


if __name__ == "__main__":
    demo_european_option()
    demo_american_option()
    demo_bermudan_option()
    demo_european_option_vector()
