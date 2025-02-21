"""
examples_runner.py
"""

import sys

from ._utils._print_utils import (
    print_runner_subtitle,
    print_runner_title,
)
from .containers import (
    example_european_option,
    example_european_option_vector,
    example_market_env,
    example_stock,
)

from .financial_calculations import (
    example_perpetual_put,
    example_implied_rate,
    examples_parity_bound,
)

from .difference_methods import (
    example_finite_diff_technique,
    example_finite_diff_greeks,
)

from .model_envs.bsm_environment import view_bsm


def run_container_examples() -> None:
    """Run the container-related examples (European Option, Stock, Market Env, etc.)."""
    print_runner_title("Container Examples")

    print_runner_subtitle("European Option")
    example_european_option()

    print_runner_subtitle("European Option Vector")
    example_european_option_vector()

    print_runner_subtitle("Market Environment")
    example_market_env()

    print_runner_subtitle("Stock")
    example_stock()


def run_financial_calculation_examples() -> None:
    """Run the examples involving financial calculations (perpetual put, implied rate, etc.)."""
    print_runner_title("Financial Calculations Examples")

    print_runner_subtitle("Perpetual Put")
    example_perpetual_put()

    print_runner_subtitle("Implied Rate & Parity Bound")
    example_implied_rate()
    examples_parity_bound()


def run_difference_method_examples() -> None:
    """Run examples that showcase difference methods."""
    print_runner_title("Finite Difference Method Examples")

    print_runner_subtitle("Basic Finite Difference Technique")
    example_finite_diff_technique()

    print_runner_subtitle("Finite Difference Greeks")
    example_finite_diff_greeks()


def main() -> None:
    """
    Main entry point for running all examples.
    """
    # run_container_examples()
    # run_financial_calculation_examples()
    # run_difference_method_examples()
    view_bsm()


if __name__ == "__main__":
    sys.exit(main())
