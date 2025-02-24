#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
demo_parity_implied_rate.py
===========================
Demonstration of using the implied_rate function from the parity_implied_rate module.

This demo calculates the implied risk-free rate based on observed call and put prices,
and displays the input parameters along with the computed rate.
"""

from src.financial_calculations import implied_rate
from demos._utils._print_utils import (
    print_title,
    print_subtitle,
    print_info,
    print_blank_line,
)


def demo_parity_implied_rate() -> None:
    """
    Demonstrate the usage of the implied_rate function.
    """
    print_title("Parity Implied Rate Demo")

    call_price = 0.5287
    put_price = 6.7143
    S = 100.0
    K = 110.0
    T = 0.5
    q = 0.01
    eps = 1e-6
    max_iter = 100

    print_subtitle("Input Parameters")
    print_info(f"  Call Price:         {call_price}")
    print_info(f"  Put Price:          {put_price}")
    print_info(f"  Underlying (S):     {S}")
    print_info(f"  Strike (K):         {K}")
    print_info(f"  Maturity (T):       {T}")
    print_info(f"  Dividend Yield (q): {q}")

    r_est = implied_rate(
        call_price, put_price, S, K, T, q=q, eps=eps, max_iter=max_iter
    )
    print_info(f"Computed Implied Risk-Free Rate: {r_est:.4f}")
    print_blank_line()


if __name__ == "__main__":
    demo_parity_implied_rate()
