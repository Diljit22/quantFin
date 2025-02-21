"""
market_env_example.py
=====================

Demonstration of creating and using the MarketEnvironment class.
"""

from src.market.market_environment import MarketEnvironment
from demo._utils._print_utils import (
    print_title,
    print_subtitle,
    print_info,
    print_blank_line,
)


def example_market_env() -> None:
    """
    Demonstrate usage of MarketEnvironment.
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
    example_market_env()
