#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
underlyings package
===================
This module provides a container for an equity underlying, allowing dynamic
updates to key parameters such as the spot price, volatility, and dividend yield.
Thread-safety is enforced using a reentrant lock. Optional properties for
discrete dividend amounts and payment times are also provided.

Classes
-------
Stock
    A thread-safe container for equity underlyings (stocks for now).
"""

from .stock import Stock

__all__ = ["Stock"]
