#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
market package
==============
This package provides a container for market-related data (namely rate).

Classes
-------
MarketEnvironment
    A lightweight, thread-safe container for market data (e.g., risk-free rate).
"""

from .market_environment import MarketEnvironment

__all__ = ["MarketEnvironment"]
