"""
finite_diff_technique.py
========================

General fallback technique class that inherits from:
  - BaseTechnique (for the abstract interface)
  - GreekMixin (for finite-difference Greeks)

It implements a default .price(...) method and an implied_volatility(...) method
that caches results. If a specialized technique does not override .delta(), etc.,
the finite-difference approach from GreekMixin is used automatically.
"""

from typing import Any, Dict
from scipy.optimize import brentq

from src.techniques.base_technique import BaseTechnique
from src.mixins.greek_mixin import GreekMixin


class FiniteDifferenceTechnique(BaseTechnique, GreekMixin):
    """
    A universal fallback technique that uses:
      - A .price(...) implementation.
      - Mixin-provided finite differences for all Greeks.
      - A bracket-based implied volatility solver with local caching.

    Caching
    -------
    - `_iv_cache` for implied vol is separate from the base caching for price.
    """

    def __init__(self, cache_results: bool = False, parallel: bool = False) -> None:
        """
        Parameters
        ----------
        cache_results : bool
            If True, enable caching in the BaseTechnique for repeated calls.
        parallel : bool
            If True, the finite-difference code in GreekMixin uses multiple threads.
        """
        BaseTechnique.__init__(self, cache_results=cache_results)
        GreekMixin.__init__(self, parallel=parallel)
        self._iv_cache: Dict[Any, float] = {}

    # ------------------------------
    # Mandatory abstract methods
    # ------------------------------
    def price(self, instrument, underlying, model, market_env, **kwargs) -> float:
        """
        Price the instrument. By default, a minimal BSM-like snippet,
        or override with your own logic.
        """
        pass

    def implied_volatility(
        self,
        instrument: Any,
        underlying: Any,
        model: Any,
        market_env: Any,
        target_price: float,
        **kwargs,
    ) -> float:
        """
        Bracket-based search for implied volatility around .price(...).
        Fallback to secant if bracket doesn't enclose zero.
        """
        # Build a unique cache key
        cache_key = self._make_cache_key

        if cache_key in self._iv_cache:
            return self._iv_cache[cache_key]

        if instrument.maturity is None or instrument.maturity <= 1e-14:
            raise ValueError(
                "Cannot compute IV for an almost expired (or invalid) option."
            )
        if target_price < 0:
            raise ValueError("Market price cannot be negative for IV calculation.")

        # bracket volatility in [1e-9, 5.0]
        low_vol, high_vol = 1e-9, 5.0
        tol = kwargs.get("tol", 1e-7)
        max_iter = kwargs.get("max_iter", 200)
        initial_guess = kwargs.get("initial_guess", 0.2)

        # Define price difference function
        def price_diff(sigma_val: float) -> float:
            # Temporarily override underlying's volatility or model's volatility
            original_sigma = getattr(underlying, "volatility", None)
            if original_sigma is None and hasattr(model, "sigma"):
                original_sigma = model.sigma
            self._set_vol(underlying, model, sigma_val)
            p = self.price(instrument, underlying, model, market_env)
            self._set_vol(underlying, model, original_sigma)
            return p - target_price

        f_low = price_diff(low_vol)
        f_high = price_diff(high_vol)

        if f_low * f_high < 0.0:
            # have a bracket; use brentq
            iv_est = brentq(price_diff, low_vol, high_vol, xtol=tol, maxiter=max_iter)
        else:
            # Fallback to secant
            iv_est = self._secant_iv(price_diff, initial_guess, tol, max_iter)

        self._iv_cache[cache_key] = iv_est
        return iv_est

    def delta(self, instrument, underlying, model, market_env, **kwargs) -> float:
        """
        By default, call GreekMixin's finite-difference delta(...) unless overridden.
        """
        return GreekMixin.delta(
            self, instrument, underlying, model, market_env, **kwargs
        )

    def gamma(self, instrument, underlying, model, market_env, **kwargs) -> float:
        """
        By default, call GreekMixin's finite-difference gamma(...) unless overridden.
        """
        return GreekMixin.gamma(
            self, instrument, underlying, model, market_env, **kwargs
        )

    def vega(self, instrument, underlying, model, market_env, **kwargs) -> float:
        return GreekMixin.vega(
            self, instrument, underlying, model, market_env, **kwargs
        )

    def theta(self, instrument, underlying, model, market_env, **kwargs) -> float:
        return GreekMixin.theta(
            self, instrument, underlying, model, market_env, **kwargs
        )

    def rho(self, instrument, underlying, model, market_env, **kwargs) -> float:
        return GreekMixin.rho(self, instrument, underlying, model, market_env, **kwargs)

    @staticmethod
    def _set_vol(underlying: Any, model: Any, new_vol: float) -> None:
        """
        Helper to temporarily set volatility in the underlying or model.
        """
        if hasattr(underlying, "volatility"):
            underlying.volatility = new_vol
        elif hasattr(model, "sigma"):
            model.sigma = new_vol

    @staticmethod
    def _secant_iv(fn, x0: float, tol: float, max_iter: int) -> float:
        """
        Simple secant fallback if bracket search fails.
        """
        x1 = x0 + 0.1
        fx0 = fn(x0)
        for _ in range(max_iter):
            fx1 = fn(x1)
            if abs(fx1) < tol:
                return x1
            denom = fx1 - fx0
            if abs(denom) < 1e-14:
                break
            x2 = x1 - fx1 * (x1 - x0) / denom
            x0, x1, fx0 = x1, x2, fx1

        return x1
