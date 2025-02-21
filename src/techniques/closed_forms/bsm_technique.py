"""
bsm_technique.py
================

Defines a closed-form Black-Scholes-Merton (BSM) technique for pricing
European options and computing their Greeks. Inherits from BaseTechnique
and implements individual methods for price, delta, gamma, vega, theta, rho,
as well as implied_volatility using a bracket-based root finder.

Caching
-------
- The `_iv_cache` dictionary stores implied vol results keyed by (spot, strike,
  maturity, option_type, market_price) to avoid repeated root searches.

"""

import math
import scipy.special
import scipy.stats
from scipy.optimize import brentq
from typing import Any, Dict, Tuple

from src.underlyings.stock import Stock
from src.market.market_environment import MarketEnvironment
from src.instruments.base_option import BaseOption
from src.models.base_model import BaseModel
from src.techniques.base_technique import BaseTechnique


def _compute_d1_d2(
    S: float, K: float, T: float, r: float, q: float, sigma: float
) -> Tuple[float, float]:
    """
    Compute d1 and d2 for the Black-Scholes formulas.

    Returns
    -------
    Tuple[float, float]
        (d1, d2) computed as:
        d1 = (ln(S/K) + (r - q + 0.5*sigma²) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
    """
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def bs_call_price(
    S: float, K: float, T: float, r: float, q: float, sigma: float
) -> float:
    """
    Black-Scholes formula for a European call option.
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(0, S - K)

    d1, d2 = _compute_d1_d2(S, K, T, r, q, sigma)
    return (S * math.exp(-q * T) * scipy.special.ndtr(d1)) - (
        K * math.exp(-r * T) * scipy.special.ndtr(d2)
    )


def bs_put_price(
    S: float, K: float, T: float, r: float, q: float, sigma: float
) -> float:
    """Black-Scholes formula for a European put option."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(0, K - S)

    d1, d2 = _compute_d1_d2(S, K, T, r, q, sigma)
    return (K * math.exp(-r * T) * scipy.special.ndtr(-d2)) - (
        S * math.exp(-q * T) * scipy.special.ndtr(-d1)
    )


def call_delta(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """
    Delta for a European call option.
    """
    if T <= 1e-14 or sigma <= 1e-14:
        return 1 if S > K else 0
    d1, _ = _compute_d1_d2(S, K, T, r, q, sigma)
    return math.exp(-q * T) * scipy.special.ndtr(d1)


def put_delta(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """
    Delta for a European put option.
    """
    if T <= 1e-14 or sigma <= 1e-14:
        return -1 if K > S else 0
    d1, _ = _compute_d1_d2(S, K, T, r, q, sigma)
    return math.exp(-q * T) * (scipy.special.ndtr(d1) - 1)


def bs_gamma(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """
    Gamma for a European option.
    """
    if T <= 1e-14 or sigma <= 1e-14:
        return 0
    d1, _ = _compute_d1_d2(S, K, T, r, q, sigma)
    return math.exp(-q * T) * scipy.stats.norm.pdf(d1) / (S * sigma * math.sqrt(T))


def bs_vega(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """
    Vega for a European option.
    """
    if T <= 1e-14 or sigma <= 1e-14:
        return 0
    d1, _ = _compute_d1_d2(S, K, T, r, q, sigma)
    return S * math.exp(-q * T) * math.sqrt(T) * scipy.stats.norm.pdf(d1)


def call_theta(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """
    Theta (per year) for a European call option.
    """
    if T <= 1e-14 or sigma <= 1e-14:
        return 0
    d1, d2 = _compute_d1_d2(S, K, T, r, q, sigma)
    term1 = (S * math.exp(-q * T) * scipy.stats.norm.pdf(d1) * sigma) / (
        2.0 * math.sqrt(T)
    )
    term2 = q * S * math.exp(-q * T) * scipy.special.ndtr(d1)
    term3 = r * K * math.exp(-r * T) * scipy.special.ndtr(d2)
    return term2 - term1 - term3


def put_theta(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """
    Theta (per year) for a European put option.
    """
    if T <= 1e-14 or sigma <= 1e-14:
        return 0
    d1, d2 = _compute_d1_d2(S, K, T, r, q, sigma)
    term1 = (S * math.exp(-q * T) * scipy.stats.norm.pdf(d1) * sigma) / (
        2.0 * math.sqrt(T)
    )
    term2 = q * S * math.exp(-q * T) * scipy.special.ndtr(-d1)
    term3 = r * K * math.exp(-r * T) * scipy.special.ndtr(-d2)
    return term3 - term1 - term2


def call_rho(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """
    Rho for a European call option.
    """
    if T <= 1e-14:
        return 0
    _, d2 = _compute_d1_d2(S, K, T, r, q, sigma)
    return K * T * math.exp(-r * T) * scipy.special.ndtr(d2)


def put_rho(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """
    Rho for a European put option.
    """
    if T <= 1e-14:
        return 0
    _, d2 = _compute_d1_d2(S, K, T, r, q, sigma)
    return -K * T * math.exp(-r * T) * scipy.special.ndtr(-d2)


class BlackScholesMertonTechnique(BaseTechnique):
    """
    Closed-form BSM technique for European calls/puts.

    - price(...) => uses the direct call/put formulas
    - delta, gamma, vega, theta, rho => direct partial derivatives
    - implied_volatility(...) => bracket-based search with fallback
    """

    def __init__(self, cache_results: bool = False) -> None:
        super().__init__(cache_results)
        # keyed by (S, K, T, opt_type, market_price)
        self._iv_cache: Dict[Any, float] = {}

    def price(
        self,
        instrument: BaseOption,
        underlying: Stock,
        model: BaseModel,
        market_env: MarketEnvironment,
    ) -> float:
        """
        Price a European call/put under Black-Scholes-Merton.
        """
        strike = instrument.strike
        maturity = instrument.maturity
        opt_type = instrument.option_type
        spot = underlying.spot
        sigma = underlying.volatility
        div = underlying.dividend
        rate = market_env.rate

        if opt_type == "Call":
            return bs_call_price(spot, strike, maturity, rate, div, sigma)
        elif opt_type == "Put":
            return bs_put_price(spot, strike, maturity, rate, div, sigma)
        else:
            raise ValueError("Unknown option_type. Must be 'Call' or 'Put'.")

    def implied_volatility(
        self,
        instrument: BaseOption,
        underlying: Stock,
        model: BaseModel,
        market_env: MarketEnvironment,
        target_price: float,
        **kwargs,
    ) -> float:
        """
        Solve for implied volatility using a bracket-based search (Brent's),
        fallback to local search if function doesn't bracket.

        target_price: the observed market price we want to match
        kwargs: optional 'tol', 'max_iter', 'initial_guess',
        """
        # Define a unique key for caching
        strike = instrument.strike
        maturity = instrument.maturity
        opt_type = instrument.option_type
        spot = underlying.spot

        cache_key = (spot, strike, maturity, opt_type, target_price)
        if cache_key in self._iv_cache:
            return self._iv_cache[cache_key]

        div = underlying.dividend
        rate = market_env.rate

        # Basic checks
        if maturity <= 1e-14:
            raise ValueError("Cannot compute IV for an almost expired option.")
        if target_price < 0:
            raise ValueError("Market price cannot be negative for IV calc.")

        # Price difference function
        def price_diff(sigma_val: float) -> float:
            if opt_type == "Call":
                return (
                    bs_call_price(spot, strike, maturity, rate, div, sigma_val)
                    - target_price
                )
            else:
                return (
                    bs_put_price(spot, strike, maturity, rate, div, sigma_val)
                    - target_price
                )

        low_vol, high_vol = 1e-9, 5.0
        tol = kwargs.get("tol", 1e-7)
        max_iter = kwargs.get("max_iter", 100)

        f_low = price_diff(low_vol)
        f_high = price_diff(high_vol)

        if f_low * f_high < 0:
            iv_est = brentq(price_diff, low_vol, high_vol, xtol=tol, maxiter=max_iter)
        else:
            # fallback to local secant near an 'initial_guess'
            def secant_method(f, x0, x1, tol_, max_it):
                f0 = f(x0)
                for _ in range(max_it):
                    f1 = f(x1)
                    if abs(f1) < tol_:
                        return x1
                    denom = f1 - f0
                    if abs(denom) < 1e-14:
                        break
                    x_next = x1 - f1 * ((x1 - x0) / denom)
                    x0, x1 = x1, x_next
                    f0 = f1
                return x1

            initial_guess = kwargs.get("initial_guess", 0.2)
            iv_est = secant_method(
                price_diff, initial_guess, initial_guess + 0.1, tol, max_iter
            )

        # store result in local cache
        self._iv_cache[cache_key] = iv_est
        return iv_est

    def delta(
        self,
        instrument: BaseOption,
        underlying: Stock,
        model: BaseModel,
        market_env: MarketEnvironment,
    ) -> float:
        strike = instrument.strike
        maturity = instrument.maturity
        opt_type = instrument.option_type
        spot = underlying.spot
        sigma = underlying.volatility
        div = underlying.dividend
        rate = market_env.rate

        if opt_type == "Call":
            return call_delta(spot, strike, maturity, rate, div, sigma)
        elif opt_type == "Put":
            return put_delta(spot, strike, maturity, rate, div, sigma)
        else:
            raise ValueError("Unknown option type for delta()")

    def gamma(
        self,
        instrument: BaseOption,
        underlying: Stock,
        model: BaseModel,
        market_env: MarketEnvironment,
    ) -> float:
        strike = instrument.strike
        maturity = instrument.maturity
        spot = underlying.spot
        sigma = underlying.volatility
        div = underlying.dividend
        rate = market_env.rate
        return bs_gamma(spot, strike, maturity, rate, div, sigma)

    def vega(
        self,
        instrument: BaseOption,
        underlying: Stock,
        model: BaseModel,
        market_env: MarketEnvironment,
    ) -> float:
        strike = instrument.strike
        maturity = instrument.maturity
        spot = underlying.spot
        sigma = underlying.volatility
        div = underlying.dividend
        rate = market_env.rate
        return bs_vega(spot, strike, maturity, rate, div, sigma)

    def theta(
        self,
        instrument: BaseOption,
        underlying: Stock,
        model: BaseModel,
        market_env: MarketEnvironment,
    ) -> float:
        strike = instrument.strike
        maturity = instrument.maturity
        opt_type = instrument.option_type
        spot = underlying.spot
        sigma = underlying.volatility
        div = underlying.dividend
        rate = market_env.rate

        if opt_type == "Call":
            return call_theta(spot, strike, maturity, rate, div, sigma)
        else:
            return put_theta(spot, strike, maturity, rate, div, sigma)

    def rho(
        self,
        instrument: BaseOption,
        underlying: Stock,
        model: BaseModel,
        market_env: MarketEnvironment,
    ) -> float:
        strike = instrument.strike
        maturity = instrument.maturity
        opt_type = instrument.option_type
        spot = underlying.spot
        sigma = underlying.volatility
        div = underlying.dividend
        rate = market_env.rate
        if opt_type == "Call":
            return call_rho(spot, strike, maturity, rate, div, sigma)
        else:
            return put_rho(spot, strike, maturity, rate, div, sigma)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(iv_cache_size={len(self._iv_cache)})"
