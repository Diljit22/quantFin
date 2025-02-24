"""
fd_bsm_jump.py
==============

Defines a closed–form Black–Scholes–Merton Jump Diffusion technique for pricing
European options and computing their Greeks. It uses Merton’s formulation in which the
option price is given by an infinite sum of Black–Scholes prices with adjusted drift and volatility.
In practice, the sum is truncated at a sufficiently high n.

The pricing function is vectorized in the jump–number dimension for speed.
Greeks and implied volatility are computed via finite differences.

Caching
-------
- The _iv_cache dictionary stores implied vol results keyed by (spot, strike, T, opt_type, market_price)
  to avoid repeated root searches.

Usage Example:
--------------
    fd_bsmj = FD_BSMJ(lam=0.5, kappa=-0.1, delta_j=0.2, cache_results=True)
    price = fd_bsmj.price(instrument, underlying, model, market_env)
    greeks = fd_bsmj.greeks(instrument, underlying, model, market_env)
    iv = fd_bsmj.implied_volatility(instrument, underlying, model, market_env, market_price=price)
    print("Price =", price)
    print("Greeks =", greeks)
    print("Implied Volatility =", iv)
"""

import math
import numpy as np
from scipy.stats import norm
from scipy.special import factorial
from typing import Any, Dict

from src.techniques.finite_diff_technique import FiniteDifferenceTechnique
from src.underlyings.stock import Stock
from src.market.market_environment import MarketEnvironment
from src.instruments.base_option import BaseOption
from src.models.base_model import BaseModel


def vectorized_bs_jump_call_price(
    S: float,
    K: float,
    T: float,
    r: float,
    div: float,
    vol: float,
    lam: float,
    kappa: float,
    delta_j: float,
    Nmax: int = 50,
) -> float:
    """
    Vectorized Black–Scholes–Merton Jump Diffusion Call Price.

    The price is given by:

        C(S,T) = sum_{n=0}^{Nmax} p_n * BS_call(S, K, T, r_n, vol_n)

    where
        p_n = exp(-lam*T) (lam*T)^n / n!
        r_n = r - lam*kappa + (n * ln(1+kappa))/T
        vol_n = sqrt(vol^2 + (n*delta_j^2)/T)
        BS_call uses the forward formulation:
            F = S_adj * exp((r_n - div)*T)   with S_adj = S - PV(dividends) (here we assume continuous div yield)
            d1 = (ln(F/K) + 0.5*vol_n^2*T)/(vol_n*sqrt(T))
            d2 = d1 - vol_n*sqrt(T)
            Price = exp(-r_n*T)*(F*N(d1)-K*N(d2))

    For simplicity, we assume S, K, T, r, div, and vol are scalars and that discrete dividends have already been
    incorporated (or div is the continuous dividend yield).

    Parameters:
      S      : float, underlying spot.
      K      : float, strike.
      T      : float, time to maturity.
      r      : float, risk–free rate.
      div    : float, continuous dividend yield.
      vol    : float, diffusion volatility.
      lam    : float, jump intensity.
      kappa  : float, expected percentage jump (E[Y-1]).
      delta_j: float, jump volatility.
      Nmax   : int, maximum jump count to sum over.

    Returns:
      float: the call price.
    """
    # Create array of jump counts
    n_arr = np.arange(Nmax + 1)  # shape (Nmax+1,)
    # Weights for each jump term
    weight = np.exp(-lam * T) * (lam * T) ** n_arr / factorial(n_arr)

    # Adjusted risk–free rate and volatility for each term.
    # Here, we adjust the drift: r_n = r - lam*kappa + (n*ln(1+kappa))/T
    # and volatility: vol_n = sqrt(vol^2 + (n*delta_j^2)/T)
    # (Assuming kappa = E[Y-1])
    r_n = r - lam * kappa + (n_arr * np.log(1 + kappa)) / T
    vol_n = np.sqrt(vol**2 + (n_arr * delta_j**2) / T)

    # Compute the forward price for each term.
    # We assume S_adj = S (or adjust externally) and use continuous dividend yield.
    F = S * np.exp((r_n - div) * T)  # note: r_n is vectorized here
    # Compute d1 and d2
    d1 = (np.log(F / K) + 0.5 * vol_n**2 * T) / (vol_n * np.sqrt(T))
    d2 = d1 - vol_n * np.sqrt(T)

    # Compute Black call price for each jump term.
    BS_call = np.exp(-r_n * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))

    # Sum weighted terms.
    return float(np.sum(weight * BS_call))


class FD_BSMJ(FiniteDifferenceTechnique):
    """
    Closed–form Black–Scholes–Merton Jump Diffusion technique for European options.

    Prices a European call (and via put–call parity, a put) using the Merton jump diffusion
    closed form. The option price is computed as an (infinite) sum of Black–Scholes prices with
    adjusted parameters. In practice, the sum is truncated at a sufficiently high number of terms.

    Greeks are computed via finite differences, and implied volatility is computed via a bracket-based search.

    Caching:
      - The _iv_cache dictionary stores implied vol results keyed by (S, K, T, opt_type, market_price).
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
        Price a European call/put option under Merton's Jump Diffusion model.

        The option price is given by:

            Price = sum_{n=0}^{Nmax} [exp(-lam*T) (lam*T)^n/n!] * BS_call(S, K, T, r_n, vol_n)

        where r_n and vol_n are adjusted as described in Merton's model.
        For puts, we use put-call parity.
        """
        lam = model.lam
        kappa = model.kappa
        delta_j = model.delta_j
        strike = instrument.strike
        T = instrument.maturity
        opt_type = instrument.option_type
        S = underlying.spot
        vol = underlying.volatility
        div = underlying.dividend
        r = market_env.rate

        # For simplicity, we assume dividends are incorporated via the continuous dividend yield.
        # (An extension could use discrete dividends similar to FD_Black.)
        call_price = vectorized_bs_jump_call_price(
            S, strike, T, r, div, vol, lam, kappa, delta_j, Nmax=50
        )
        if opt_type == "Call":
            return call_price
        elif opt_type == "Put":
            # Put price from put-call parity: Call - exp(-rT)*(F - K)
            F = S * math.exp((r - div) * T)
            put_price = call_price - math.exp(-r * T) * (F - strike)
            return put_price
        else:
            raise ValueError("Unknown option_type. Must be 'Call' or 'Put'.")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(iv_cache_size={len(self._iv_cache)})"
