"""
parity_bounds_example.py
========================

Demonstration of using utility functions from parity_bounds.py.

Functions demonstrated:
- put_call_parity: Computes the complementary option price via put-call parity.
- put_call_bound: Computes naive lower and upper bounds for an option price.
- lower_bound_rate: Computes a lower bound on the risk-free rate.
"""

from src.financial_calculations.parity_bounds import (
    put_call_parity,
    put_call_bound,
    lower_bound_rate,
)
from demo._utils._print_utils import (
    print_title,
    print_subtitle,
    print_info,
    print_blank_line,
)


def example_put_call_parity() -> None:
    option_price = 6.71
    S = 100.0
    K = 110.0
    r = 0.08
    T = 0.5
    q = 0.01

    print_subtitle("put_call_parity Example")
    print_info("Input Parameters:")
    print_info(f"  Option Price (Put): {option_price}")
    print_info(f"  Underlying (S): {S}")
    print_info(f"  Strike (K): {K}")
    print_info(f"  Risk-free Rate (r): {r}")
    print_info(f"  Maturity (T): {T}")
    print_info(f"  Dividend Yield (q): {q}")

    call_price = put_call_parity(option_price, S, K, r, T, q=q, price_call=False)
    print_info(f"Computed Call Price: {call_price:.4f}")
    print_blank_line()


def example_put_call_bound() -> None:
    S = 36.0
    K = 37.0
    r = 0.055
    T = 0.5

    print_subtitle("put_call_bound Example")
    print_info("Input Parameters:")
    print_info(f"  Underlying (S): {S}")
    print_info(f"  Strike (K): {K}")
    print_info(f"  Risk-free Rate (r): {r}")
    print_info(f"  Maturity (T): {T}")

    lb, ub = put_call_bound(S, K, r, T, bound_call=False)
    print_info("Computed Bounds:")
    print_info(f"  Lower Bound: {lb:.4f}")
    print_info(f"  Upper Bound: {ub:.4f}")
    print_blank_line()


def example_lower_bound_rate() -> None:
    call_price = 0.5287
    put_price = 6.7143
    S = 100.0
    K = 110.0
    T = 0.5

    print_subtitle("lower_bound_rate Example")
    print_info("Input Parameters:")
    print_info(f"  Call Price: {call_price}")
    print_info(f"  Put Price:  {put_price}")
    print_info(f"  Underlying (S): {S}")
    print_info(f"  Strike (K): {K}")
    print_info(f"  Maturity (T): {T}")

    rate_bound = lower_bound_rate(call_price, put_price, S, K, T)
    print_info(f"Computed Lower Bound on r: {rate_bound:.4f}")
    print_blank_line()


def examples_parity_bound() -> None:
    """
    Run all parity bounds examples.
    """
    print_title("Parity Bounds Examples")

    example_put_call_parity()
    example_put_call_bound()
    example_lower_bound_rate()


if __name__ == "__main__":
    examples_parity_bound()
