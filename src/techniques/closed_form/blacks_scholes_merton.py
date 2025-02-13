# techniques/closed_form/blacks_scholes_merton.py

"""
BlackScholesMertonTechnique
===========================

Implements a closed-form solution for European options under the
Black-Scholes-Merton framework. Inherits from BaseTechnique to provide:

- price(...)
- implied_volatility(...)
- greeks: delta(...), gamma(...), vega(...), theta(...), rho(...)

This technique relies on:
- The Model (e.g., BlackScholesMertonModel) for parameters:
  sigma, risk-free rate r, dividend yield q, etc.
- A MarketEnvironment for the spot price (S) and discounting or
  additional checks if needed.
- A standard European Option instrument with attributes:
  * strike
  * maturity
  * option_type ('call' or 'put')
  * option_style ('european')

Performance & HPC
-----------------
1. Vectorization:
   - For large-scale usage (e.g., many strikes), consider NumPy vectorized
     calls. Here, we demonstrate a scalar approach.
2. Parallelization:
   - Methods could be parallelized if computing prices/Greeks for many instruments.
3. Caching:
   - We implement a simple in-memory cache for implied volatility to speed up repeated queries.
4. For especially tight loops or massive scale, consider numba.jit or a Cython implementation.

Examples
--------
>>> from src.instruments.option import Option
>>> from src.models.blacks_scholes_merton_model import BlackScholesMertonModel
>>> from src.market.market_environment import MarketEnvironment
>>> from techniques.closed_form.blacks_scholes_merton import BlackScholesMertonTechnique

>>> # Setup a MarketEnvironment
>>> env = MarketEnvironment(spot_prices={"AAPL": 150.0})
>>> # Setup a BSM model
>>> bsm_model = BlackScholesMertonModel(sigma=0.20, risk_free_rate=0.03, dividend_yield=0.01)
>>> # Create a call option
>>> call_option = Option("AAPL", maturity=1.0, strike=150.0, option_type="call")
>>> # Price the option
>>> technique = BlackScholesMertonTechnique()
>>> price = technique.price(call_option, bsm_model, env)
>>> print(price)
12.345678  # hypothetical result

>>> # Compute implied volatility given a market price
>>> iv = technique.implied_volatility(call_option, bsm_model, env, market_price=12.0)
>>> print(iv)
0.198  # approximate implied vol
"""

import math
import functools
from typing import Any, Dict

from src.techniques.base_technique import BaseTechnique
from src.instruments.base_instrument import BaseInstrument
from src.models.base_model import BaseModel
from src.market.market_environment import MarketEnvironment


# ------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------
def _phi(x: float) -> float:
    """
    Standard normal PDF.
    """
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _Phi(z: float) -> float:
    """
    Standard normal CDF using an error function approximation.
    For HPC or extreme precision, consider a more advanced implementation or
    direct calls to scipy.special. This is a typical approximation.
    """
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _bs_call_price(
    S: float, K: float, T: float, r: float, q: float, sigma: float
) -> float:
    """
    Black-Scholes formula for a European call option.

    Parameters
    ----------
    S : float
        Spot price.
    K : float
        Strike price.
    T : float
        Time to maturity (in years).
    r : float
        Risk-free interest rate.
    q : float
        Continuous dividend yield.
    sigma : float
        Volatility.

    Returns
    -------
    float
        The theoretical price of a European call option under BSM.
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        # Edge cases: return max(0, S - K) if T ~ 0, ignoring discounting
        return max(0.0, (S - K))

    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    # discounted factors
    disc1 = math.exp(-q * T) * _Phi(d1)
    disc2 = math.exp(-r * T) * _Phi(d2)

    return S * disc1 - K * disc2


def _bs_put_price(
    S: float, K: float, T: float, r: float, q: float, sigma: float
) -> float:
    """
    Black-Scholes formula for a European put option.
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(0.0, (K - S))

    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    disc1 = math.exp(-q * T) * _Phi(-d1)
    disc2 = math.exp(-r * T) * _Phi(-d2)

    return K * disc2 - S * disc1


def _bs_call_delta(
    S: float, K: float, T: float, r: float, q: float, sigma: float
) -> float:
    """
    Black-Scholes delta for a call (w.r.t. spot).
    """
    if T <= 1e-14 or sigma <= 1e-14:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return math.exp(-q * T) * _Phi(d1)


def _bs_put_delta(
    S: float, K: float, T: float, r: float, q: float, sigma: float
) -> float:
    """
    Black-Scholes delta for a put (w.r.t. spot).
    """
    if T <= 1e-14 or sigma <= 1e-14:
        return -1.0 if K > S else 0.0
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return math.exp(-q * T) * (_Phi(d1) - 1.0)


def _bs_gamma(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """
    Gamma is the same for both calls and puts in BSM.
    """
    if T <= 1e-14 or sigma <= 1e-14:
        return 0.0
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return math.exp(-q * T) * _phi(d1) / (S * sigma * math.sqrt(T))


def _bs_vega(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """
    Vega is the same for both calls and puts in BSM. It's the partial
    derivative of price w.r.t. sigma, typically expressed in points per 1% change.
    """
    if T <= 1e-14 or sigma <= 1e-14:
        return 0.0
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return S * math.exp(-q * T) * math.sqrt(T) * _phi(d1)


def _bs_call_theta(
    S: float, K: float, T: float, r: float, q: float, sigma: float
) -> float:
    """
    Theta for a call, partial derivative w.r.t. time. Typically negative.
    Returned as 'per year' convention (i.e., not dividing by 365).
    """
    if T <= 1e-14 or sigma <= 1e-14:
        # approximate limit
        return 0.0

    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    first_term = -(S * math.exp(-q * T) * _phi(d1) * sigma / (2.0 * math.sqrt(T)))
    second_term = q * S * math.exp(-q * T) * _Phi(d1)
    third_term = r * K * math.exp(-r * T) * _Phi(d2)

    return first_term - second_term - third_term


def _bs_put_theta(
    S: float, K: float, T: float, r: float, q: float, sigma: float
) -> float:
    """
    Theta for a put, partial derivative w.r.t. time. Same notes as call_theta.
    """
    if T <= 1e-14 or sigma <= 1e-14:
        return 0.0

    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    first_term = -(S * math.exp(-q * T) * _phi(d1) * sigma / (2.0 * math.sqrt(T)))
    second_term = -q * S * math.exp(-q * T) * _Phi(-d1)
    third_term = r * K * math.exp(-r * T) * _Phi(-d2)

    return first_term + second_term - third_term


def _bs_call_rho(
    S: float, K: float, T: float, r: float, q: float, sigma: float
) -> float:
    """
    Rho for a call (partial derivative of price w.r.t. interest rate).
    """
    if T <= 1e-14:
        return 0.0
    d2 = (math.log(S / K) + (r - q - 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return K * T * math.exp(-r * T) * _Phi(d2)


def _bs_put_rho(
    S: float, K: float, T: float, r: float, q: float, sigma: float
) -> float:
    """
    Rho for a put.
    """
    if T <= 1e-14:
        return 0.0
    d2 = (math.log(S / K) + (r - q - 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return -K * T * math.exp(-r * T) * _Phi(-d2)


# ------------------------------------------------------------------------
# The Technique Class
# ------------------------------------------------------------------------
class BlackScholesMertonTechnique(BaseTechnique):
    """
    A closed-form Black-Scholes-Merton technique for pricing European calls/puts,
    as well as computing Greeks and implied volatility. Leverages HPC-friendly
    design by allowing for potential vectorization or parallelization if needed.

    Attributes
    ----------
    _iv_cache : Dict[str, float]
        A simple in-memory cache for implied volatilities, keyed by an internal
        string or tuple representing the option contract. This speeds up repeated
        queries for the same instrument & observed price.
    """

    def __init__(self) -> None:
        super().__init__()
        # Simple dictionary cache: {("symbol", strike, maturity, "option_type", market_price): implied_vol}
        self._iv_cache: Dict[Any, float] = {}

    def price(
        self, instrument: BaseInstrument, model: BaseModel, market_env: Any, **kwargs
    ) -> float:
        """
        Compute the fair value of a European option under the BSM model.

        Parameters
        ----------
        instrument : BaseInstrument
            The option to be priced (must have 'strike', 'maturity', 'option_type').
        model : BaseModel
            The BlackScholesMertonModel providing sigma, r, q, etc.
        market_env : Any
            Market environment or data. We fetch spot from this environment.
        **kwargs :
            Additional parameters (unused by default).

        Returns
        -------
        float
            The option price.
        """
        # Extract data from instrument
        option = instrument
        strike = option.strike
        maturity = option.maturity
        opt_type = option.option_type.lower()  # "call" or "put"

        # Check if the option style is 'european'
        if getattr(option, "option_style", "european").lower() != "european":
            raise ValueError(
                "BlackScholesMertonTechnique only supports European-style options."
            )

        # Extract data from model
        params = model.get_params()
        sigma = params["sigma"]
        r = params["r"]
        q = params["dividend_yield"]

        # Fetch the spot price from MarketEnvironment
        if isinstance(market_env, MarketEnvironment):
            S = market_env.get_spot_price(option.underlying_symbol)
        else:
            raise ValueError(
                "market_env must be a MarketEnvironment or have equivalent get_spot_price method."
            )

        # Compute price via closed-form formula
        if opt_type == "call":
            return _bs_call_price(S, strike, maturity, r, q, sigma)
        elif opt_type == "put":
            return _bs_put_price(S, strike, maturity, r, q, sigma)
        else:
            raise ValueError("Unknown option_type. Must be 'call' or 'put'.")

    def implied_volatility(
        self,
        instrument: BaseInstrument,
        model: BaseModel,
        market_env: Any,
        market_price: float,
        **kwargs
    ) -> float:
        """
        Compute implied volatility given an observed market price for a European call/put.

        Parameters
        ----------
        instrument : BaseInstrument
            The option for which we want the implied volatility.
        model : BaseModel
            The BSM model from which to get initial guesses or references to r, q, etc.
        market_env : Any
            The environment from which to fetch spot, if needed.
        market_price : float
            The observed market price of the option.
        **kwargs : dict, optional
            Additional parameters (e.g., initial vol guess, tolerance, max iterations).

        Returns
        -------
        float
            The implied volatility (annualized).
        """
        # Check cache first
        cache_key = (
            instrument.underlying_symbol,
            instrument.option_type,
            instrument.strike,
            instrument.maturity,
            market_price,
        )
        if cache_key in self._iv_cache:
            return self._iv_cache[cache_key]

        # Extract data
        S = market_env.get_spot_price(instrument.underlying_symbol)
        strike = instrument.strike
        maturity = instrument.maturity

        params = model.get_params()
        r = params["r"]
        q = params["dividend_yield"]

        # Edge cases
        if maturity <= 0:
            raise ValueError(
                "Cannot compute implied volatility for expired or zero-maturity option."
            )
        if market_price < 0:
            raise ValueError("Market price cannot be negative.")

        # For a call or put payoff
        is_call = instrument.option_type.lower() == "call"

        # Boundaries for implied vol searching
        vol_lower = 1e-9
        vol_upper = 5.0  # 500% vol upper bound for extreme cases
        tol = kwargs.get("tol", 1e-7)
        max_iter = kwargs.get("max_iter", 100)
        # Optional initial guess
        guess = kwargs.get("initial_guess", 0.2)

        def price_diff(vol: float) -> float:
            if is_call:
                px = _bs_call_price(S, strike, maturity, r, q, vol)
            else:
                px = _bs_put_price(S, strike, maturity, r, q, vol)
            return px - market_price

        # If a local Newton or secant method is desired, we can do that.
        # We'll implement Brent's for robustness:
        from math import isclose

        def brent_bisection(f, a, b, tolerance, maxiter):
            fa = f(a)
            fb = f(b)
            if abs(fa) < tolerance:
                return a
            if abs(fb) < tolerance:
                return b
            if fa * fb > 0:
                # fallback to simple approach: the function does not bracket the root
                return a if abs(fa) < abs(fb) else b

            for _ in range(maxiter):
                mid = 0.5 * (a + b)
                fm = f(mid)
                if abs(fm) < tolerance:
                    return mid
                if fa * fm < 0:
                    b = mid
                    fb = fm
                else:
                    a = mid
                    fa = fm
            return 0.5 * (a + b)

        # We start by checking if guess is OK or not:
        # We'll do a small step around guess to bracket if possible:
        test_low = price_diff(vol_lower)
        test_high = price_diff(vol_upper)

        if test_low * test_high > 0:
            # If they have the same sign, fallback to a local approach
            # We'll just do a local approach around guess using a simple newton or secant

            def secant(f, x0, x1, tol, maxiter):
                f0 = f(x0)
                for _ in range(maxiter):
                    f1 = f(x1)
                    if abs(f1) < tol:
                        return x1
                    if isclose(x1, x0, rel_tol=1e-14):
                        break
                    d = (x1 - x0) / (f1 - f0)
                    x0, x1 = x1, x1 - f1 * d
                    f0 = f1
                return x1

            iv_est = secant(price_diff, guess, guess + 0.1, tol, max_iter)
        else:
            iv_est = brent_bisection(price_diff, vol_lower, vol_upper, tol, max_iter)

        # Store in cache
        self._iv_cache[cache_key] = iv_est
        return iv_est

    def delta(
        self, instrument: BaseInstrument, model: BaseModel, market_env: Any, **kwargs
    ) -> float:
        """
        Compute Delta (dPrice/dSpot).

        Returns
        -------
        float
            The option Delta.
        """
        # Extract data
        S = market_env.get_spot_price(instrument.underlying_symbol)
        strike = instrument.strike
        T = instrument.maturity

        params = model.get_params()
        r = params["r"]
        q = params["dividend_yield"]
        sigma = params["sigma"]

        # Edge checks
        if T <= 0:
            return (
                1.0
                if (instrument.option_type.lower() == "call" and S > strike)
                else 0.0
            )

        if instrument.option_type.lower() == "call":
            return _bs_call_delta(S, strike, T, r, q, sigma)
        else:  # put
            return _bs_put_delta(S, strike, T, r, q, sigma)

    def gamma(
        self, instrument: BaseInstrument, model: BaseModel, market_env: Any, **kwargs
    ) -> float:
        """
        Compute Gamma (d^2Price/dSpot^2).

        Returns
        -------
        float
            The option Gamma.
        """
        # Extract data
        S = market_env.get_spot_price(instrument.underlying_symbol)
        strike = instrument.strike
        T = instrument.maturity

        params = model.get_params()
        r = params["r"]
        q = params["dividend_yield"]
        sigma = params["sigma"]

        return _bs_gamma(S, strike, T, r, q, sigma)

    def vega(
        self, instrument: BaseInstrument, model: BaseModel, market_env: Any, **kwargs
    ) -> float:
        """
        Compute Vega (dPrice/dVol).

        Returns
        -------
        float
            The option Vega.
        """
        S = market_env.get_spot_price(instrument.underlying_symbol)
        strike = instrument.strike
        T = instrument.maturity

        params = model.get_params()
        r = params["r"]
        q = params["dividend_yield"]
        sigma = params["sigma"]

        return _bs_vega(S, strike, T, r, q, sigma)

    def theta(
        self, instrument: BaseInstrument, model: BaseModel, market_env: Any, **kwargs
    ) -> float:
        """
        Compute Theta (dPrice/dTime). Typically negative for long options.

        Returns
        -------
        float
            The option Theta (annualized).
        """
        S = market_env.get_spot_price(instrument.underlying_symbol)
        strike = instrument.strike
        T = instrument.maturity

        params = model.get_params()
        r = params["r"]
        q = params["dividend_yield"]
        sigma = params["sigma"]

        if instrument.option_type.lower() == "call":
            return _bs_call_theta(S, strike, T, r, q, sigma)
        else:
            return _bs_put_theta(S, strike, T, r, q, sigma)

    def rho(
        self, instrument: BaseInstrument, model: BaseModel, market_env: Any, **kwargs
    ) -> float:
        """
        Compute Rho (dPrice/dRate).

        Returns
        -------
        float
            The option Rho.
        """
        S = market_env.get_spot_price(instrument.underlying_symbol)
        strike = instrument.strike
        T = instrument.maturity

        params = model.get_params()
        r = params["r"]
        q = params["dividend_yield"]
        sigma = params["sigma"]

        if instrument.option_type.lower() == "call":
            return _bs_call_rho(S, strike, T, r, q, sigma)
        else:
            return _bs_put_rho(S, strike, T, r, q, sigma)

    @property
    def name(self):
        return "ClosedFormBSM"
