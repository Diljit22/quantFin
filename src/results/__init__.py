#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
results package
===============
This package contains classes and functions for storing and processing
option evaluation results.

Modules
-------
option_evaluation
    Contains the OptionEvaluation dataclass which encapsulates the pricing model,
    evaluation technique, computed option price, option Greeks, implied volatility,
    and additional instrument and underlying data.
"""

from .option_evaluation import OptionEvaluation

__all__ = ["OptionEvaluation"]
