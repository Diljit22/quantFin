"""
perpetual_put_example.py
========================

Demonstration of pricing a perpetual put option using the perpetual_put function.
"""

from src.financial_calculations.perpetual_put import perpetual_put
from demo._utils._print_utils import (
    print_title,
    print_subtitle,
    print_info,
    print_blank_line,
)


def example_perpetual_put() -> None:
    """
    Demonstrate usage of the perpetual_put function.
    """
    print_title("Perpetual Put Option Example")

    S = 150.0
    K = 100.0
    r = 0.08
    vol = 0.2
    q = 0.005

    print_subtitle("Input Parameters")
    print_info(f"  Stock Price (S):     {S}")
    print_info(f"  Strike Price (K):    {K}")
    print_info(f"  Risk-Free Rate (r):  {r}")
    print_info(f"  Volatility (vol):    {vol}")
    print_info(f"  Dividend Yield (q):  {q}")

    value = perpetual_put(S, K, r, vol, q)
    print_info(f"Computed Perpetual Put Value: {value:.10f}")
    print_blank_line()


if __name__ == "__main__":
    example_perpetual_put()
