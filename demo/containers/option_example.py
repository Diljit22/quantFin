"""
option_example.py
=================

Demonstration of creating and using EuropeanOption and EuropeanOptionVector classes.
"""

import numpy as np

from src.instruments.european_option import EuropeanOption
from src.instruments.european_option_vector import EuropeanOptionVector
from demo._utils._print_utils import (
    print_title,
    print_subtitle,
    print_info,
    print_blank_line,
)


def example_european_option() -> None:
    """
    Demonstrate usage of the EuropeanOption class.
    """
    print_title("European Option Example")

    print_subtitle("Creating a European Call Option")
    euro_call = EuropeanOption(strike=100.0, maturity=1.0, is_call=True)
    print_info(f"Created European Call Option:\n{euro_call}")

    payoff_high = euro_call.payoff(120.0)
    payoff_low = euro_call.payoff(90.0)
    print_info(f"Payoff at spot=120.0: {payoff_high}")
    print_info(f"Payoff at spot=90.0: {payoff_low}")

    print_info(f"Option Type: {euro_call.option_type}")

    print_subtitle("Creating an Updated Strike (110.0)")
    new_euro_call = euro_call.with_strike(110.0)
    print_info(f"European Call with updated strike:\n{new_euro_call}")
    print_info(
        f"Payoff at spot=120.0 for updated strike: {new_euro_call.payoff(120.0)}"
    )

    print_subtitle("Creating an Updated Maturity (2.0 years)")
    updated_maturity_option = euro_call.with_maturity(2.0)
    print_info(f"European Call with updated maturity:\n{updated_maturity_option}")

    print_subtitle("Creating a Companion Option (Inverted Call/Put)")
    companion = euro_call.companion_option
    print_info(f"Companion option:\n{companion}")
    print_info(f"Companion payoff at spot=90.0: {companion.payoff(90.0)}")

    print_blank_line()


def example_european_option_vector() -> None:
    """
    Demonstrate usage of the EuropeanOptionVector class.
    """
    print_title("European Option Vector Example")

    print_subtitle("Creating a Vectorized European Call")
    strikes = np.array([90.0, 100.0, 110.0])
    euro_vector = EuropeanOptionVector(strikes=strikes, maturity=1.0, is_call=True)
    print_info(f"Created Option Vector:\n{euro_vector}")

    payoff_scalar = euro_vector.payoff(120.0)
    print_info("Payoff at spot=120.0 (broadcast):")
    print_info(str(payoff_scalar))

    print_info(f"Option Type: {euro_vector.option_type}")

    print_subtitle("Updating Strike Array to [110.0, 115.0]")
    new_strikes = np.array([110.0, 115.0])
    new_vector = euro_vector.with_strike(new_strikes)
    print_info(f"New Vector:\n{new_vector}")
    print_info("Vectorized payoff at spot=130.0:")
    print_info(str(new_vector.payoff(130.0)))

    print_subtitle("Updating Maturity to 1.5 Years")
    updated_maturity_vector = euro_vector.with_maturity(1.5)
    print_info(
        f"European Option Vector with updated maturity:\n{updated_maturity_vector}"
    )

    print_subtitle("Creating Companion Option Vector (Inverted Call/Put)")
    companion_vector = euro_vector.companion_option
    print_info(f"Companion Vector:\n{companion_vector}")
    print_info("Companion payoff at spot=90.0:")
    print_info(str(companion_vector.payoff(90.0)))

    print_blank_line()


if __name__ == "__main__":
    example_european_option()
    example_european_option_vector()
