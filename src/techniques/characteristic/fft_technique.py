"""
fourier_pricing_technique.py
============================

Defines the FourierPricingTechnique, a subclass of FiniteDifferenceTechnique that prices options
using an FFT-based Carr–Madan pricer (prFFT approach). This technique leverages the FFT algorithm
to efficiently compute option prices on a log-strike grid and then interpolates the result at the desired
strike. Greeks (delta, gamma, vega, etc.) are computed via finite-difference approximations.

Usage:
    Create an instance of FourierPricingTechnique and use its methods to price an option,
    compute its Greeks, or solve for implied volatility.
"""

import math
import numpy as np
import scipy.integrate
import scipy.optimize
from typing import Any, Callable, Tuple
import warnings

# Import the base finite-difference technique (which includes GreekMixin functionality)
from src.techniques.finite_diff_technique import FiniteDifferenceTechnique
from src.underlyings.stock import Stock
from src.market.market_environment import MarketEnvironment
from src.instruments.base_option import BaseOption as Instrument

# --------------------------------------------------------------------------
# Supporting functions for FFT pricing
# --------------------------------------------------------------------------


def isATM(S: float, K: float, eps: float = 0.01) -> bool:
    """Return True if the option is at-the-money (|S-K| <= eps)."""
    return abs(S - K) <= eps


def genFuncs(
    phi: Callable[[complex], complex],
    S: float,
    K: float,
    alpha: float,
    disc: float,
    eps: float = 0.01,
):
    """
    Generate functions required for FFT pricing.

    Returns:
      dampener : function to damp the integrand (ensures square-integrability)
      twiPhi   : transformed characteristic function used in the Fourier transform.
    """
    atmFlag = isATM(S, K, eps)
    if atmFlag:
        dampener = lambda x: np.exp(alpha * x)
        denom_ = lambda u: alpha**2 + alpha - u**2 + 1j * u * (2 * alpha + 1)
        twiPhi = lambda u: phi(u - 1j * (1 + alpha)) / denom_(u)
    else:
        dampener = lambda x: np.sinh(alpha * x)
        ft = (
            lambda u: 1 / (1j * u + 1)
            - 1 / (disc * 1j * u)
            - phi(u - 1j) / (u**2 - 1j * u)
        )
        twiPhi = lambda u: (ft(u - 1j * alpha) - ft(u + 1j * alpha)) / 2
    return dampener, twiPhi


# --------------------------------------------------------------------------
# FourierPricingTechnique Class Definition
# --------------------------------------------------------------------------


class FourierPricingTechnique(FiniteDifferenceTechnique):
    """
    Pricing technique that computes the option price using an FFT-based Carr–Madan pricer.

    The FFT approach computes the price over a log-strike grid and then interpolates to obtain
    the price at a desired strike. Greeks (delta, gamma, etc.) are computed using finite-difference
    approximations.
    """

    def __init__(
        self,
        alpha: float = 1.5,
        trunc: int = 7,
        n: int = 12,
        cache_results: bool = False,
    ) -> None:
        """
        Initialize the FourierPricingTechnique.

        Parameters
        ----------
        alpha : float, optional
            Dampening parameter (default is 1.0).
        trunc : int, optional
            Exponent for the truncation limit (B = 2**trunc).
        n : int, optional
            Determines the number of FFT points L = 2**n.
        cache_results : bool, optional
            If True, enables caching for repeated computations.
        """
        super().__init__(cache_results=cache_results)
        self.alpha = float(alpha)
        self.trunc = trunc
        self.n = n

    def _fft_price(
        self,
        char_func: Callable[[complex], complex],
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        call: bool = True,
        ATMeps: float = 0.01,
        normalize: bool = False,
        debug: bool = False,
    ) -> Tuple[float, Tuple[np.ndarray, np.ndarray]]:
        """
        Compute the option price via FFT using the prFFT approach.

        Parameters
        ----------
        char_func : callable
            The characteristic function φ(u) of the log–price.
        S : float
            Underlying asset price.
        K : float
            Reference strike price.
        T : float
            Time to maturity.
        r : float
            Risk-free rate.
        q : float
            Dividend yield.
        call : bool, optional
            True for call option; False for put (default is True).
        ATMeps : float, optional
            Tolerance for determining at-the-money (default is 0.01).
        normalize : bool, optional
            If True, normalizes the underlying to 1 and rescales results (default is False).
        debug : bool, optional
            If True, prints intermediate debug information.

        Returns
        -------
        Tuple containing the price at the reference strike and a tuple (K_grid, values).
        """
        S_orig = S
        if normalize:
            if debug:
                print("Normalizing: Original S =", S_orig)
            S = 1.0
            K = K / S_orig
            if debug:
                print("Normalized strike K =", K)
        k_ref = np.log(K)
        disc = math.exp(-r * T)
        if debug:
            print("k_ref (log(K)) =", k_ref)
            print("Discount factor disc =", disc)
        dampener, twi = genFuncs(char_func, S, K, self.alpha, disc, eps=ATMeps)
        damp_kref = dampener(k_ref)
        if debug:
            print("dampener(k_ref) =", damp_kref)
        # Set up Fourier integration grid.
        dy = 2 ** (self.trunc - self.n)
        B = 2**self.trunc
        L = 2**self.n
        if debug:
            print("Integration grid parameters: dy =", dy, "B =", B, "L =", L)
        mul = disc * (B / math.pi) / damp_kref
        if debug:
            print("Scaling multiplier (mul) =", mul)
        Y = np.arange(0, B, dy)
        if debug:
            print("First 5 values of Y =", Y[:5])
        Q = np.exp(-1j * k_ref * Y) * twi(Y)
        Q[0] /= 2
        if debug:
            print("First 5 values of Q =", Q[:5])
        fft_result = np.fft.ifft(Q)
        values = mul * np.real(fft_result)
        values = np.fft.fftshift(values)
        if debug:
            print("First 5 FFT output values after fftshift =", values[:5])
        delta_k = 2 * math.pi / B
        k_grid = k_ref + (np.arange(L) - L / 2) * delta_k
        K_grid = np.exp(k_grid)
        if debug:
            print("delta_k =", delta_k)
            print("First 5 values of k_grid =", k_grid[:5])
            print("First 5 values of K_grid (before rescaling) =", K_grid[:5])
        if normalize:
            K_grid = S_orig * K_grid
            if debug:
                print("Rescaled K_grid (first 5) =", K_grid[:5])
        if not call:
            values = values - S * math.exp(-q * T) + K * disc
        # Interpolate to obtain the price at the reference strike.
        ref_target = K if not normalize else S_orig * K
        idx = np.argmin(np.abs(K_grid - ref_target))
        price_ref = values[idx]
        if normalize:
            price_ref *= S_orig
        return price_ref, (K_grid, values)

    def price(
        self,
        instrument: Any,
        underlying: Stock,
        model: Any,
        market_env: MarketEnvironment,
    ) -> float:
        """
        Compute the option price using the FFT pricer.

        Parameters are extracted from the instrument, underlying, model, and market environment.
        """
        S = underlying.spot
        K = instrument.strike
        T = instrument.maturity
        r = market_env.rate
        q = underlying.dividend
        call_flag = instrument.option_type.lower() == "call"
        phi = model.characteristic_function(T, S, r, q)
        price_val, _ = self._fft_price(
            phi,
            S,
            K,
            T,
            r,
            q,
            call=call_flag,
            ATMeps=0.01,
            normalize=False,
            debug=False,
        )
        return price_val

    def delta(
        self,
        instrument: Any,
        underlying: Stock,
        model: Any,
        market_env: MarketEnvironment,
    ) -> float:
        """
        Compute the option delta via a central finite-difference approximation.
        """
        S0 = underlying.spot
        h = 1e-2 * S0 if S0 != 0 else 1e-2
        underlying_up = Stock(S0 + h, underlying.volatility, underlying.dividend)
        underlying_down = Stock(S0 - h, underlying.volatility, underlying.dividend)
        price_up = self.price(instrument, underlying_up, model, market_env)
        price_down = self.price(instrument, underlying_down, model, market_env)
        delta_val = (price_up - price_down) / (2 * h)
        return delta_val

    def gamma(
        self,
        instrument: Any,
        underlying: Stock,
        model: Any,
        market_env: MarketEnvironment,
        step: float = None,
        **kwargs,
    ) -> float:
        """
        Compute Gamma (the second derivative with respect to the underlying spot)
        using a five-point finite-difference approximation.
        """
        S0 = underlying.spot
        if step is None:
            step = 1e-2 * S0 if S0 != 0 else 1e-2
        # Create new Stock instances for perturbations.
        stock_p2 = Stock(S0 + 2 * step, underlying.volatility, underlying.dividend)
        stock_p1 = Stock(S0 + step, underlying.volatility, underlying.dividend)
        stock_m1 = Stock(S0 - step, underlying.volatility, underlying.dividend)
        stock_m2 = Stock(S0 - 2 * step, underlying.volatility, underlying.dividend)
        f0 = self.price(instrument, underlying, model, market_env)
        f1 = self.price(instrument, stock_p1, model, market_env)
        f_1 = self.price(instrument, stock_m1, model, market_env)
        f2 = self.price(instrument, stock_p2, model, market_env)
        f_2 = self.price(instrument, stock_m2, model, market_env)
        gamma_val = (-f2 + 16 * f1 - 30 * f0 + 16 * f_1 - f_2) / (12 * step**2)
        return gamma_val

    def implied_volatility(
        self,
        instrument: Any,
        underlying: Stock,
        model: Any,
        market_env: MarketEnvironment,
        target_price: float,
        tol: float = 1e-7,
        max_iter: int = 100,
        **kwargs,
    ) -> float:
        """
        Solve for the implied volatility by reinitializing the model with each volatility guess.
        """
        from scipy.optimize import brentq

        def price_diff(sigma_val: float) -> float:
            if hasattr(model, "with_volatility") and callable(model.with_volatility):
                new_model = model.with_volatility(sigma_val)
            else:
                raise NotImplementedError(
                    "Model must implement a 'with_volatility' method."
                )
            p = self.price(instrument, underlying, new_model, market_env, **kwargs)
            return p - target_price

        low_vol, high_vol = 1e-9, 5.0
        iv = brentq(price_diff, low_vol, high_vol, xtol=tol, maxiter=max_iter)
        return iv

    def vega(
        self,
        instrument: Any,
        underlying: Stock,
        model: Any,
        market_env: MarketEnvironment,
        step: float = None,
        **kwargs,
    ) -> float:
        """
        Compute Vega (sensitivity to volatility) using a finite difference approach.
        """
        sigma = getattr(underlying, "volatility", None)
        if sigma is None:
            sigma = getattr(model, "sigma", None)
        if sigma is None or sigma <= 0:
            return 0.0
        if step is None:
            step = 1e-2 * sigma

        base_args = {
            "instrument": instrument,
            "underlying": underlying,
            "model": model,
            "market_env": market_env,
            "sigma": sigma,
        }

        def param_wrapper(**kwargs2):
            kwargs2.pop("instrument", None)
            kwargs2.pop("underlying", None)
            kwargs2.pop("market_env", None)
            kwargs2.pop("model", None)
            param_sigma = kwargs2.pop("sigma", None)
            if param_sigma is not None:
                if hasattr(model, "with_volatility") and callable(
                    model.with_volatility
                ):
                    new_model = model.with_volatility(param_sigma)
                else:
                    raise NotImplementedError(
                        "Model must implement a 'with_volatility' method."
                    )
            else:
                new_model = model
            result = self.get_price(
                instrument, underlying, new_model, market_env, **kwargs2
            )
            return result

        return self._finite_diff_1st(param_wrapper, base_args, "sigma", step)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(alpha={self.alpha}, trunc={self.trunc}, n={self.n}, cache_results={self._cache_results})"
