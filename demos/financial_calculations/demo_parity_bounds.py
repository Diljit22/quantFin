#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
demo_parity_bounds.py
========================
Demonstration of using utility functions from the parity_bounds module.

This demo illustrates how to compute:
  - The complementary option price via put_call_parity.
  - Option price bounds using put_call_bound.
  - A lower bound on the risk-free rate using lower_bound_rate.
"""

from src.financial_calculations import (
    put_call_parity,
    put_call_bound,
    lower_bound_rate,
)
from demos._utils._print_utils import (
    print_title,
    print_subtitle,
    print_info,
    print_blank_line,
)


def demo_put_call_parity() -> None:
    """
    Demonstrate usage of the put_call_parity function.

    Computes the complementary call price from a given put price using:
        C - P = S * exp(-q * T) - K * exp(-r * T)
    """
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


def demo_put_call_bound() -> None:
    """
    Demonstrate usage of the put_call_bound function.

    Computes the lower and upper bounds for an option price based on the
    put-call inequalities.
    """
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

    lower_bound, upper_bound = put_call_bound(S, K, r, T, bound_call=False)
    print_info("Computed Bounds:")
    print_info(f"  Lower Bound: {lower_bound:.4f}")
    print_info(f"  Upper Bound: {upper_bound:.4f}")
    print_blank_line()


def demo_lower_bound_rate() -> None:
    """
    Demonstrate usage of the lower_bound_rate function.

    Computes a lower bound on the risk-free rate from observed call and put prices.
    """
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


def run_parity_bounds_examples() -> None:
    """
    Run all parity bounds examples.

    Sequentially executes the examples for:
      - put_call_parity,
      - put_call_bound, and
      - lower_bound_rate.
    """
    print_title("Parity Bounds Examples")
    demo_put_call_parity()
    demo_put_call_bound()
    demo_lower_bound_rate()


if __name__ == "__main__":
    run_parity_bounds_examples()
