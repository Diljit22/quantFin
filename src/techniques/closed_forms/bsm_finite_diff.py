"""
bsm_finite_diff.py
================

Defines a closed-form Black-Scholes-Merton (BSM) technique for pricing
European options and computing their Greeks. Inherits from FiniteDifferenceTechnique.

Caching
-------
- The `_iv_cache` dictionary stores implied vol results keyed by (spot, strike,
  maturity, option_type, market_price) to avoid repeated root searches.

"""

import scipy.stats
from typing import Any, Dict

from src.techniques.closed_forms.bsm_technique import bs_put_price, bs_call_price
from src.techniques.finite_diff_technique import FiniteDifferenceTechnique
from src.underlyings.stock import Stock
from src.market.market_environment import MarketEnvironment
from src.instruments.base_option import BaseOption
from src.models.base_model import BaseModel


class FD_BSM(FiniteDifferenceTechnique):
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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(iv_cache_size={len(self._iv_cache)})"
