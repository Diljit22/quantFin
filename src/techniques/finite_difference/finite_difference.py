# src/techniques/finite_difference/finite_difference.py

"""
Finite-Difference Technique
===========================
Provides a universal finite-difference fallback for pricing and Greek
calculations, plus an implied-volatility routine, all in a single file.

Contents
--------
1) GreekMixin:
   - A mixin that provides general-purpose finite-difference methods for
     computing Greeks (delta, gamma, vega, theta, rho), or higher-order
     sensitivities, by perturbing the relevant parameters.
   - Includes an LRU cache to optimize repeated pricing calls.

2) FiniteDiffTechnique(BaseTechnique, GreekMixin):
   - A concrete technique class that:
     * Has a `_price(...)` method for a single set of params. By default,
       it can do a fallback snippet or call the child's override.
     * Provides implied_volatility(...) via root-finding around `price(...)`.
     * Inherits from `GreekMixin` for finite-diff delta, gamma, etc.
     * For HPC contexts, optional parallelization using `concurrent.futures`
       or joblib can be toggled for repeated finite-difference calls.

Usage Example
-------------
>>> from src.techniques.finite_difference.finite_difference import FiniteDiffTechnique
>>> from src.instruments.option import Option
>>> from src.market.market_environment import MarketEnvironment
>>> from src.models.base_model import BaseModel   # or a child
>>> env = MarketEnvironment(spot_prices={"AAPL": 150.0})
>>> my_model = MyCustomModel(sigma=0.2, r=0.03, q=0.01)
>>> opt = Option("AAPL", maturity=1.0, strike=150.0, option_type="call")

>>> tech = FiniteDiffTechnique(use_parallel=True)

>>> px = tech.price(opt, my_model, env)
>>> print("Price:", px)
>>> dlt = tech.delta(opt, my_model, env)
>>> print("Delta:", dlt)
"""

import math
import concurrent.futures
from functools import lru_cache
from typing import Dict, Any

import numpy as np

from src.techniques.base_technique import BaseTechnique
from src.techniques.finite_difference.greek_mixin import GreekMixin
from src.instruments.base_instrument import BaseInstrument
from src.models.base_model import BaseModel
from src.market.market_environment import MarketEnvironment


class FiniteDiffTechnique(BaseTechnique, GreekMixin):
    """
    A general finite-difference fallback for computing:
     - price(...) : single shot price
     - implied_volatility(...) : bracket & root-find around price
     - delta, gamma, vega, theta, rho : via GreekMixin (central differences)

    Notes
    -----
    - `_price(...)` is a per-parameter-set logic used by the finite-diff
      engine. We do a fallback snippet but typically the child technique
      overrides it for custom logic (CharFunctionIntegration, PDE, etc.).
    - HPC concurrency can be toggled with use_parallel.
    """

    def __init__(self, use_parallel: bool = False):
        super().__init__()
        GreekMixin.__init__(self, parallel=use_parallel)

    # ----------------------------------------------------------------
    # BaseTechnique interface
    # ----------------------------------------------------------------
    def price(
        self, instrument: BaseInstrument, model: BaseModel, market_env: Any, **kwargs
    ) -> float:
        """
        Price the instrument by building param dict & calling cached_price(...).
        """
        params = self._build_params_dict(instrument, model, market_env)
        return self.cached_price(**params)

    def implied_volatility(
        self,
        instrument: BaseInstrument,
        model: BaseModel,
        market_env: Any,
        market_price: float,
        **kwargs
    ) -> float:
        """
        Compute implied volatility by root-finding around price(...).

        We bracket in [1e-9..5.0]. If f(vol_lower)*f(vol_upper) > 0,
        fallback to secant. Otherwise do bisection.
        """
        orig_sigma = model._params["sigma"]
        tol = kwargs.get("tol", 1e-7)
        max_iter = kwargs.get("max_iter", 100)
        vol_lower, vol_upper = 1e-9, 5.0
        initial_guess = kwargs.get("initial_guess", 0.2)

        def f(vol: float) -> float:
            old_sigma = model.get_params().get("sigma", 0.0)
            self._set_model_param(model, "sigma", vol)
            p = self.price(instrument, model, market_env)
            self._set_model_param(model, "sigma", old_sigma)
            return p - market_price

        fl = f(vol_lower)
        fu = f(vol_upper)
        # revert model
        model._params["sigma"] = orig_sigma

        if fl * fu > 0:
            return self._secant_iv(f, initial_guess, tol, max_iter)
        else:
            return self._bisection_iv(f, vol_lower, vol_upper, tol, max_iter)

    # ----------------------------------------------------------------
    # Greeks using finite differences
    # ----------------------------------------------------------------
    def delta(
        self, instrument: BaseInstrument, model: BaseModel, market_env: Any, **kwargs
    ) -> float:
        """
        Finite-diff Delta wrt Spot.
        """
        base_params = self._build_params_dict(instrument, model, market_env)
        step = kwargs.get("step", 1e-3 * base_params["spot"])
        return self.finite_difference_greek("spot", base_params, step, order=1)

    def gamma(
        self, instrument: BaseInstrument, model: BaseModel, market_env: Any, **kwargs
    ) -> float:
        """
        Finite-diff Gamma wrt Spot^2.
        """
        base_params = self._build_params_dict(instrument, model, market_env)
        step = kwargs.get("step", 1e-3 * base_params["spot"])
        return self.finite_difference_greek("spot", base_params, step, order=2)

    def vega(
        self, instrument: BaseInstrument, model: BaseModel, market_env: Any, **kwargs
    ) -> float:
        """
        Finite-diff Vega wrt sigma.
        """
        base_params = self._build_params_dict(instrument, model, market_env)
        sigma = base_params.get("sigma", 0.0)
        if sigma <= 0.0:
            return 0.0
        step = kwargs.get("step", 1e-3 * sigma)
        return self.finite_difference_greek("sigma", base_params, step, order=1)

    def theta(
        self, instrument: BaseInstrument, model: BaseModel, market_env: Any, **kwargs
    ) -> float:
        """
        Finite-diff Theta wrt maturity (T).
        """
        base_params = self._build_params_dict(instrument, model, market_env)
        if base_params["maturity"] <= 1e-12:
            return 0.0
        step = kwargs.get("step", 1e-4)
        return self.finite_difference_greek("maturity", base_params, step, order=1)

    def rho(
        self, instrument: BaseInstrument, model: BaseModel, market_env: Any, **kwargs
    ) -> float:
        """
        Finite-diff Rho wrt interest rate r.
        """
        base_params = self._build_params_dict(instrument, model, market_env)
        step = kwargs.get("step", 1e-4)
        return self.finite_difference_greek("r", base_params, step, order=1)

    # ----------------------------------------------------------------
    # The internal _price(...) for finite differences
    # ----------------------------------------------------------------
    def _price(self, **params) -> float:
        """
        Default fallback if child hasn't overridden. Does a trivial BSM snippet
        The recommended approach is to override this in the child
        technique if you want a custom approach (CharFunc, PDE, etc.).
        """
        # parse param dictionary
        S = params["spot"]
        K = params["strike"]
        T = params["maturity"]
        r = params["r"]
        q = params["q"]
        sigma = params["sigma"]
        otype = params["option_type"].lower()

        if T < 1e-12:
            payoff = (S - K) if (otype == "call") else (K - S)
            return max(0.0, payoff)

        # minimal BSM snippet
        from math import log, sqrt, exp, erf

        def _Phi(x: float) -> float:
            return 0.5 * (1.0 + erf(x / (math.sqrt(2.0))))

        d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (
            sigma * math.sqrt(T)
        )
        d2 = d1 - sigma * math.sqrt(T)
        exp_q = math.exp(-q * T)
        exp_r = math.exp(-r * T)

        if otype == "call":
            return S * exp_q * _Phi(d1) - K * exp_r * _Phi(d2)
        else:
            return K * exp_r * _Phi(-d2) - S * exp_q * _Phi(-d1)

    def _build_params_dict(
        self, instrument: BaseInstrument, model: BaseModel, market_env: Any
    ) -> Dict[str, Any]:
        """
        Gather relevant parameters from instrument, model, market_env
        into a dict. We also store the 'model' in __model__ so child
        classes can do advanced logic if needed.
        """
        S = market_env.get_spot_price(instrument.underlying_symbol)
        m_params = model.get_params()
        r = m_params.get("r", 0.0)
        q = m_params.get("dividend_yield", 0.0)
        sigma = m_params.get("sigma", 0.0)

        T = instrument.maturity
        K = instrument.strike
        otype = instrument.option_type

        return {
            "spot": S,
            "strike": K,
            "maturity": T,
            "r": r,
            "q": q,
            "sigma": sigma,
            "option_type": otype,
            "__model__": model,
        }

    # ----------------------------------------------------------------
    # Helpers for implied vol
    # ----------------------------------------------------------------
    def _set_model_param(self, model: BaseModel, param_name: str, val: float):
        """
        Temporarily override a param in model._params. The user must
        handle concurrency or side effects if they call in parallel.
        """
        params = model.get_params()
        params[param_name] = val
        model._params[param_name] = val

    def _secant_iv(self, f, guess, tol, max_iter):
        """
        Fallback secant if bracket fails.
        """
        x0 = guess
        x1 = guess + 0.1
        fx0 = f(x0)
        for _ in range(max_iter):
            fx1 = f(x1)
            if abs(fx1) < tol:
                return x1
            denom = fx1 - fx0
            if abs(denom) < 1e-14:
                break
            d = (x1 - x0) / denom
            x0, x1 = x1, x1 - fx1 * d
            fx0 = fx1
        return x1

    def _bisection_iv(self, f, low, high, tol, max_iter):
        """
        Bisection approach if bracket succeeds.
        """
        fl = f(low)
        fh = f(high)
        for _ in range(max_iter):
            mid = 0.5 * (low + high)
            fm = f(mid)
            if abs(fm) < tol:
                return mid
            if fl * fm < 0:
                high = mid
                fh = fm
            else:
                low = mid
                fl = fm
        return 0.5 * (low + high)
