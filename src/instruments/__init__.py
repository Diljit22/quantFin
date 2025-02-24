#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
instruments package
===================

This package defines various option instruments including the abstract
base option and concrete implementations for American, Bermudan, and European options.
It also provides a vectorized European option for handling multiple strikes.
"""

from .base_option import BaseOption
from .american_option import AmericanOption
from .bermudan_option import BermudanOption
from .european_option import EuropeanOption
from .european_option_vector import EuropeanOptionVector

__all__ = [
    "BaseOption",
    "AmericanOption",
    "BermudanOption",
    "EuropeanOption",
    "EuropeanOptionVector",
]
