#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
financial_calculations package
===============================

This package provides utility functions for option pricing calculations, including:

- Functions from parity_bounds.py:
    - put_call_parity: Computes the complementary option price via put-call parity.
    - put_call_bound: Computes naive lower/upper bounds for call or put options.
    - lower_bound_rate: Bounds risk-free rate from below using the put-call inequality.
- Function from parity_implied_rate.py:
    - implied_rate: Solves numerically for implied risk-free rate via Brent's method.
- Function from perpetual_put.py:
    - perpetual_put: Prices a perpetual put option via a closed-form solution.

Modules
-------
parity_bounds
    Provides put_call_parity, put_call_bound, and lower_bound_rate.
parity_implied_rate
    Provides implied_rate.
perpetual_put
    Provides perpetual_put.
"""

from .parity_bounds import put_call_parity, put_call_bound, lower_bound_rate
from .parity_implied_rate import implied_rate
from .perpetual_put import perpetual_put

__all__ = [
    "put_call_parity",
    "put_call_bound",
    "lower_bound_rate",
    "implied_rate",
    "perpetual_put",
]
