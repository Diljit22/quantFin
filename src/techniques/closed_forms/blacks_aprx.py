"""
fd_black.py
===========

Defines a closed–form Black’s approximation technique for pricing European options
with discrete dividends. It inherits from FiniteDifferenceTechnique, and Greeks are
computed via finite differences.

Caching
-------
- The _iv_cache dictionary stores implied vol results keyed by (spot, strike,
  maturity, option_type, market_price) to avoid repeated root searches.
  
Usage Example:
--------------
    fd_black = FD_Black(cache_results=True)
    price = fd_black.price(instrument, underlying, model, market_env)
    print("Option Price =", price)
    Second pricing works
"""

import math
import numpy as np
from scipy.stats import norm
from typing import Any, Dict

from src.techniques.finite_diff_technique import FiniteDifferenceTechnique
from src.underlyings.stock import Stock
from src.market.market_environment import MarketEnvironment
from src.instruments.base_option import BaseOption
from src.models.base_model import BaseModel


def black_dividend_price(
    S: float,
    K: float,
    r: float,
    T: float,
    vol: float,
    q: np.ndarray,
    dYr: np.ndarray,
    call: bool = True,
) -> float:
    """
    Price a European option using Black’s approximation adjusted for discrete dividends.

    The method subtracts the present value of dividends (paid before T) from S,
    computes an adjusted forward price F, and then applies Black’s formula.

    Parameters:
      S    : float
             Current underlying price.
      K    : float
             Strike price.
      r    : float
             Risk–free interest rate.
      T    : float
             Time to maturity (years).
      vol  : float
             Volatility.
      q    : array-like
             Dividend amounts.
      dYr  : array-like
             Dividend payment times (in years). Only dividends with dYr < T are used.
      call : bool, optional (default True)
             True for a call option; False for a put option.

    Returns:
      float: Option price.
    """
    # Ensure dividends and dividend times are 1D arrays.
    q = np.asarray(q).flatten()
    dYr = np.asarray(dYr).flatten()
    # Only include dividends that occur before maturity.
    mask = dYr < T
    PV_div = np.sum(q[mask] * np.exp(-r * dYr[mask])) if np.any(mask) else 0.0

    # Adjust the underlying price.
    S_adj = S - PV_div
    # Compute forward price.
    F = S_adj * math.exp(r * T)
    if F <= 0:
        return 0.0

    d1 = (math.log(F / K) + 0.5 * vol * vol * T) / (vol * math.sqrt(T))
    d2 = d1 - vol * math.sqrt(T)

    call_price = math.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))
    if call:
        return call_price
    else:
        # Use put-call parity: Call - Put = exp(-rT)*(F - K)
        put_price = call_price - math.exp(-r * T) * (F - K)
        return put_price


def blacksApproximation(S, K, r, T, vol, q, dYr, call=True):
    """Price an American option paying discrete dividends.

    Parameters
    ----------
    S   : float : Current price of stock.
    K   : float : Strike price of the option.
    r   : float : Annualized risk-free interest rate, continuously compounded.
    T   : float : Time, in years, until maturity.
    vol : float : Volatility of the stock.
    q   : array : Dividend payment(s).
    dYr : array : Time, in years, of dividend payout(s).
    call: bool, optional : If pricing call.

    Returns
    -------
    optionValue : float : Value of option.

    Example(s)
    ----------
    >>> qdiv, qtimes = np.array([.7, .7]), np.array([[3/12, 5/12]])
    >>> blacksApproximation(S=50, K=55, r=.1, T=.5, vol=.3,
                            q=qdiv, dYr=qtimes)
    >>> 2.6756877949596003

    >>> blacksApproximation(S=55, K=50, r=.1, T=.5, vol=.3,
                            q=qdiv, dYr=qtimes, call=False)
    >>> 1.8991667424392134

    """
    dYr, div = np.append(dYr, T), np.append(q, 0)
    rateTime = r * dYr
    disc = np.exp(-rateTime)
    adjS = S - np.cumsum(disc * div)
    adjK = K * disc

    stdDev = vol * np.sqrt(dYr)
    logChange = np.log(adjS / K) + rateTime
    d1 = (logChange) / stdDev + stdDev / 2
    d2 = d1 - stdDev

    if call:
        cdf_d1, cdf_d2 = norm().cdf(d1), norm().cdf(d2)
        value = cdf_d1 * adjS - cdf_d2 * adjK

    else:
        cdf_neg_d2, cdf_neg_d1 = norm().cdf(-d2), norm().cdf(-d1)
        value = cdf_neg_d2 * adjK - cdf_neg_d1 * adjS

    return max(value)


class FD_Black(FiniteDifferenceTechnique):
    """
    Closed–form Black’s approximation technique for European options with discrete dividends.

    This technique prices an option using Black’s formula on an adjusted underlying,
    where the adjustment is the present value of dividends (provided as discrete payments).

    Greeks are computed via finite differences (inherited from FiniteDifferenceTechnique).

    Caching:
      - The _iv_cache dictionary stores implied volatility results keyed by
        (S, K, T, opt_type, market_price).
    """

    def __init__(self, cache_results: bool = False) -> None:
        super().__init__(cache_results)
        self._iv_cache: Dict[Any, float] = {}

    def price(
        self,
        instrument: BaseOption,
        underlying: Stock,
        model: BaseModel,
        market_env: MarketEnvironment,
    ) -> float:
        """
        Price a European call/put option under Black’s approximation with discrete dividends.

        The underlying is expected to have the attributes:
          - spot: current price.
          - volatility: volatility.
          - dividend: a NumPy array of dividend amounts.
          - dividend_times: a NumPy array of dividend payment times (in years).
        If dividend_times is not provided, a continuous dividend yield is assumed.

        Parameters:
          instrument: BaseOption with attributes strike, maturity, option_type.
          underlying: Stock.
          model: BaseModel (unused).
          market_env: MarketEnvironment with attribute rate.

        Returns:
          float: Option price.
        """
        strike = instrument.strike
        maturity = instrument.maturity
        opt_type = instrument.option_type
        S = underlying.spot
        vol = underlying.volatility
        r = market_env.rate

        # Use discrete dividends if available; otherwise, use continuous dividend yield.
        if hasattr(underlying, "dividend_times"):
            q = underlying.discrete_dividend
            dYr = underlying.dividend_times
        else:
            q = np.array([underlying.dividend])
            dYr = np.array([maturity])  # effectively no dividend before expiration

        if opt_type == "Call":
            return blacksApproximation(S, strike, r, maturity, vol, q, dYr, call=True)
        elif opt_type == "Put":
            return blacksApproximation(S, strike, r, maturity, vol, q, dYr, call=False)
        else:
            raise ValueError("Unknown option_type. Must be 'Call' or 'Put'.")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(iv_cache_size={len(self._iv_cache)})"
