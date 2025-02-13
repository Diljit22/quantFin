# src/techniques/characteristic/char_function_fft.py

"""
CarrMadanFFTTechnique
=====================

Implements the Carr-Madan method (Fourier transform approach) for
pricing European options via a model's characteristic function.

Key Features
------------
1) We inherit from FiniteDiffTechnique to reuse:
   - Implied vol routine (root-find)
   - Greeks: delta, gamma, vega, theta, rho (finite differences)
2) We define a 'price(...)' method that computes a single-strike
   price by performing the FFT on a log-strike grid, then
   picking the nearest strike.
3) We incorporate a dampening parameter alpha > 0 to ensure
   integrability for out-of-the-money calls (the standard approach
   from Carr & Madan).
4) HPC concurrency is omitted here, but you can adapt it if you
   compute large grids or multiple maturities.

Usage
-----
>>> from src.techniques.characteristic.char_function_fft import CarrMadanFFTTechnique
>>> from src.instruments.option import Option
>>> from src.market.market_environment import MarketEnvironment
>>> from src.models.base_model import BaseModel

>>> model = MyBSMModel(...)  # must define model.char_func(S, T)->CF
>>> env   = MarketEnvironment(spot_prices={"TEST":100.0})
>>> opt   = Option("TEST", 1.0, 100.0, "call", "european")

>>> fft_tech = CarrMadanFFTTechnique(alpha=1.5, N=2**12, B=600.0)
>>> price_val = fft_tech.price(opt, model, env)
>>> print("FFT Price:", price_val)
"""

import math
import cmath
import numpy as np
from typing import Any, Dict, Tuple

from src.techniques.finite_difference.finite_difference import FiniteDiffTechnique
from src.instruments.base_instrument import BaseInstrument
from src.models.base_model import BaseModel
from src.market.market_environment import MarketEnvironment


class CarrMadanFFTTechnique(FiniteDiffTechnique):
    """
    A Carr–Madan FFT approach for European options. Inherits from
    FiniteDiffTechnique so it automatically satisfies the abstract
    methods for Greeks (theta, rho, vega) and implied volatility
    via finite-difference or root-finding.

    Parameters
    ----------
    alpha : float, default=1.5
        Dampening parameter > 0 for calls. Typically 1..2 for stable integrals.
    N : int, default=2**12
        Number of FFT points, must be a power of 2 for efficiency.
    B : float, default=600.0
        Log-strike domain upper bound. The step in log(K) space is B / N.
    use_parallel : bool, default=False
        If you want concurrency for large transforms, adapt as needed.

    Notes
    -----
    - If you want to handle multiple strikes at once, you'd do a
      single FFT pass, store the entire array of call prices, then
      do interpolation for each strike. This class demonstrates a
      single-strike usage.
    - The user-facing price(...) does the entire transform, picks
      the nearest strike, and returns that price.
    """

    def __init__(
        self,
        alpha: float = 1.5,
        N: int = 2**12,
        B: float = 600.0,
        use_parallel: bool = False,
    ) -> None:
        """Constructor for CarrMadanFFTTechnique."""
        super().__init__(use_parallel=use_parallel)
        self.alpha = alpha
        self.N = N
        self.B = B
        self.use_parallel = use_parallel
        self._cache_iv: Dict[Any, float] = {}

        self.name = "Carr-Madan FFT"

    def price(
        self,
        instrument: BaseInstrument,
        model: BaseModel,
        market_env: MarketEnvironment,
        **kwargs
    ) -> float:
        """
        Price a single European call/put by performing the FFT over
        log-strikes and picking the nearest index to the instrument's strike.

        Parameters
        ----------
        instrument : BaseInstrument
            Must define 'strike', 'maturity', 'option_type'='call'/'put',
            style='european'.
        model : BaseModel
            Must define model.char_func(S, T) => returns phi(u).
        market_env : MarketEnvironment
            Provides the spot price, etc.
        **kwargs : dict
            Unused.

        Returns
        -------
        float
            The fair price of the option at the given strike.
        """
        if getattr(instrument, "option_style", "european").lower() != "european":
            raise ValueError("CarrMadanFFTTechnique only supports European payoffs.")

        strike = instrument.strike
        if strike is None:
            raise ValueError("Instrument must have a valid strike.")
        T = instrument.maturity
        is_call = instrument.option_type.lower() == "call"

        if not isinstance(market_env, MarketEnvironment):
            raise ValueError("Expected a MarketEnvironment for 'market_env'.")

        S = market_env.get_spot_price(instrument.underlying_symbol)

        # Model parameters
        m_params = model.get_params()
        r = m_params.get("r", 0.0)
        q = m_params.get("dividend_yield", 0.0)

        # We'll do an FFT for log-strikes in [0..B], get array of call prices.
        call_arr, strike_arr = self._compute_fft(
            model=model,
            spot=S,
            maturity=T,
            r=r,
            q=q,
            alpha=self.alpha,
            N=self.N,
            B=self.B,
        )

        ln_strikes = np.log(strike_arr)
        lnK = math.log(strike)
        idx = (np.abs(ln_strikes - lnK)).argmin()
        call_val = float(call_arr[idx])

        if is_call:
            return call_val
        else:
            # put = call - S e^{-qT} + K e^{-rT}
            disc_s = math.exp(-q * T)
            disc_k = math.exp(-r * T)
            put_val = call_val - (S * disc_s) + (strike * disc_k)
            return put_val

    def _compute_fft(
        self,
        model: BaseModel,
        spot: float,
        maturity: float,
        r: float,
        q: float,
        alpha: float,
        N: int,
        B: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform the Carr-Madan FFT approach for a grid of log-strikes.

        Parameters
        ----------
        model : BaseModel
            Must define model.char_func(S, T) => a function phi(u).
        spot : float
            Current underlying spot price.
        maturity : float
            Time to maturity in years.
        r : float
            Risk-free rate (continuous compounding).
        q : float
            Continuous dividend yield.
        alpha : float
            Dampening parameter > 0.
        N : int
            Number of FFT points, power of 2.
        B : float
            Upper bound of log-strike domain.

        Returns
        -------
        call_prices : np.ndarray
            1D array of call prices for strikes in [exp(0)..exp(B)].
        strike_grid : np.ndarray
            1D array of strike values = exp(k) for k in [0..B].
        """
        # Get the characteristic function
        cf_method = getattr(model, "char_func", None)
        if cf_method is None:
            raise NotImplementedError(
                "Model must define 'char_func(S, T)' -> CF function."
            )
        phi = cf_method(spot, maturity)  # => phi(u: complex) -> complex

        # Step in logK domain
        delta_k = B / N
        k_vals = np.arange(N) * delta_k  # [0..B]
        strike_grid = np.exp(k_vals)

        # freq domain spacing
        lam = math.pi / B
        j_indices = np.arange(N)
        u_vals = j_indices * lam

        discount = math.exp(-r * maturity)

        # Build G array
        G = np.zeros(N, dtype=complex)
        for j in range(N):
            u = u_vals[j]
            # shift = u - i(alpha+1)
            u_shift = complex(u, -(alpha + 1.0))
            cf_val = phi(u_shift)
            # denominator = alpha(alpha+1) - u^2 + i(2alpha+1)u
            # typical
            denom = alpha * (alpha + 1.0) - (u * u) + 1j * (2 * alpha + 1.0) * u
            if abs(denom) < 1e-14:
                G[j] = 0.0
            else:
                G[j] = discount * cf_val / denom

        # trapezoid weighting or simpson
        w = np.ones(N, dtype=complex)
        w[0] = 0.5  # half weight at endpoints
        # multiply
        F_in = G * w

        # do FFT
        fft_vals = np.fft.fft(F_in)
        # scale
        call_prices = np.exp(-alpha * k_vals) / math.pi * np.real(fft_vals) * lam

        return call_prices, strike_grid

    def implied_volatility(
        self,
        instrument: BaseInstrument,
        model: BaseModel,
        market_env: Any,
        market_price: float,
        **kwargs
    ) -> float:
        """
        Solve for implied volatility from a single-strike Carr-Madan approach
        by root-finding around the 'price(...)' method, changing model._params['sigma'].

        For advanced models (Heston, etc.), you'd adapt the relevant parameter.

        Returns
        -------
        float
            Implied volatility found via bracket + bisection or fallback secant.
        """
        # If you want to store results for repeated queries
        cache_key = (
            instrument.underlying_symbol,
            instrument.strike,
            instrument.maturity,
            instrument.option_type,
            market_price,
        )
        if cache_key in self._cache_iv:
            return self._cache_iv[cache_key]

        tol = kwargs.get("tol", 1e-7)
        max_iter = kwargs.get("max_iter", 100)
        vol_lower, vol_upper = 1e-9, 5.0
        guess = kwargs.get("initial_guess", 0.2)

        orig_sigma = model._params.get("sigma", 0.2)

        def froot(vol: float) -> float:
            model._params["sigma"] = vol
            val = self.price(instrument, model, market_env)
            return val - market_price

        fl = froot(vol_lower)
        fu = froot(vol_upper)
        # revert
        model._params["sigma"] = orig_sigma

        if fl * fu > 0:
            # fallback secant
            def secant(g, x0, x1, tol_, iters):
                fx0 = g(x0)
                for _ in range(iters):
                    fx1 = g(x1)
                    if abs(fx1) < tol_:
                        return x1
                    dnm = fx1 - fx0
                    if abs(dnm) < 1e-14:
                        break
                    step = (x1 - x0) / dnm
                    x0, x1 = x1, x1 - fx1 * step
                    fx0 = fx1
                return x1

            iv_est = secant(froot, guess, guess + 0.1, tol, max_iter)
        else:
            # bisection
            iv_est = self._bisection_iv(froot, vol_lower, vol_upper, tol, max_iter)

        self._cache_iv[cache_key] = iv_est
        return iv_est

    def _bisection_iv(self, f, low, high, tol, max_iter):
        """
        Simple bisection approach.
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
