#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
option_evaluation.py
====================
Defines the OptionEvaluation class, a container for option evaluation results.

This class provides a simple, immutable (frozen dataclass) container for storing
the results of option evaluations, including model name, pricing technique, computed
option price, option Greeks, implied volatility, and any additional data for the
option and underlying.
"""

from dataclasses import dataclass
from typing import Optional, Any


@dataclass(frozen=True)
class OptionEvaluation:
    """
    Container for option evaluation results.

    Parameters
    ----------
    model : str
        The name of the pricing model used (e.g., "BlackScholesMerton").
    technique : str
        The evaluation technique used (e.g., "ClosedForm", "FiniteDiff").
    price : float
        The computed option price.
    delta : Optional[float], optional
        The option delta.
    gamma : Optional[float], optional
        The option gamma.
    vega : Optional[float], optional
        The option vega.
    theta : Optional[float], optional
        The option theta.
    rho : Optional[float], optional
        The option rho.
    implied_vol : Optional[float], optional
        The computed implied volatility.
    instrument_data : Optional[Any], optional
        Additional data describing the option (e.g., strike, maturity, is_call).
    underlying_data : Optional[Any], optional
        Additional data describing the underlying (e.g., spot, dividend, volatility).
    """

    model: str
    technique: str
    price: float
    delta: Optional[float] = None
    gamma: Optional[float] = None
    vega: Optional[float] = None
    theta: Optional[float] = None
    rho: Optional[float] = None
    implied_vol: Optional[float] = None
    instrument_data: Optional[Any] = None
    underlying_data: Optional[Any] = None

    def __str__(self) -> str:
        """
        Return a formatted string representation of the option evaluation.

        Returns
        -------
        str
            A multi-line string with details on the pricing model, technique,
            computed option price, Greeks, implied volatility, and any additional data.
        """
        parts = [
            "Option Evaluation:",
            f"  Model: {self.model}",
            f"  Technique: {self.technique}",
            f"  Price: {self.price:.4f}",
        ]
        if self.delta is not None:
            parts.append(f"  Delta: {self.delta:.4f}")
        if self.gamma is not None:
            parts.append(f"  Gamma: {self.gamma:.4f}")
        if self.vega is not None:
            parts.append(f"  Vega: {self.vega:.4f}")
        if self.theta is not None:
            parts.append(f"  Theta: {self.theta:.4f}")
        if self.rho is not None:
            parts.append(f"  Rho: {self.rho:.4f}")
        if self.implied_vol is not None:
            parts.append(f"  Implied Volatility: {self.implied_vol:.4f}")
        if self.instrument_data is not None:
            parts.append(f"  Instrument Data: {self.instrument_data}")
        if self.underlying_data is not None:
            parts.append(f"  Underlying Data: {self.underlying_data}")
        return "\n".join(parts)
