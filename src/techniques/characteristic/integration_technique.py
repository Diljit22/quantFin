"""
fourier_pricing_technique.py
============================

Defines the FourierPricingTechnique, a subclass of BaseTechnique that prices options
by integrating the characteristic function (via Fourier inversion). It implements methods
for pricing, computing implied volatility, and calculating Greeks (delta, gamma, vega,
theta, rho).

The option price and delta are computed by integrating the characteristic function
using a helper method (integrate_phi), while the remaining Greeks are approximated using
finite difference methods.

Usage:
    Create an instance of FourierPricingTechnique and use its methods to price an option,
    compute its Greeks, or solve for implied volatility.
"""

import numpy as np
import scipy.integrate
import scipy.optimize
from typing import Any, Callable, Tuple

from src.underlyings.stock import Stock
from src.market.market_environment import MarketEnvironment
from src.techniques.finite_diff_technique import FiniteDifferenceTechnique
import warnings
from scipy.integrate import IntegrationWarning

# Suppress all IntegrationWarning messages
warnings.filterwarnings("ignore", category=IntegrationWarning)


class IntegrationTechnique(FiniteDifferenceTechnique):
    """
    Pricing technique that computes the option price by integrating the
    characteristic function of the log-price (via Fourier inversion).

    The integration returns both the price and the delta (first derivative with respect to spot)
    directly. The remaining Greeks (gamma, vega, theta, rho) are approximated using finite differences.
    """

    def __init__(self, cache_results: bool = False) -> None:
        """
        Initialize the FourierPricingTechnique.

        Parameters
        ----------
        cache_results : bool, optional
            If True, enables caching for repeated computations (default: False).
        """
        super().__init__(cache_results)

    @staticmethod
    def integrate_phi(
        phi: Callable[[complex], complex],
        S: float,
        K: float,
        r: float,
        T: float,
        q: float,
        call: bool = True,
    ) -> Tuple[float, float]:
        """
        Compute the option price and delta by integrating the characteristic function.

        The integration is based on Fourier inversion techniques.

        Parameters
        ----------
        phi : Callable[[complex], complex]
            The characteristic function of the log-price.
        S : float
            Current underlying spot price.
        K : float
            Strike price of the option.
        r : float
            Risk-free rate (annualized, continuously compounded).
        T : float
            Time to maturity (in years).
        q : float
            Continuous dividend yield.
        call : bool, optional
            True for a call option, False for a put (default: True).

        Returns
        -------
        Tuple[float, float]
            A tuple containing (price, delta).
        """
        # Define the "twisted" characteristic function for delta calculation.
        twPhi = lambda u: phi(u - 1j) / phi(-1j)
        k_log = np.log(K)
        # Transformation functions for integration.
        trfPhi = lambda u: np.imag(np.exp(-1j * u * k_log) * phi(u)) / u
        trfTwi = lambda u: np.imag(np.exp(-1j * u * k_log) * twPhi(u)) / u

        A, _ = scipy.integrate.quad(trfPhi, 0, np.inf)
        B, _ = scipy.integrate.quad(trfTwi, 0, np.inf)

        pITMCall = 0.5 + A / np.pi
        deltaCall = 0.5 + B / np.pi

        # Adjust for present value factors.
        adjS = S * np.exp(-q * T)
        adjK = K * np.exp(-r * T)

        if call:
            price = adjS * deltaCall - adjK * pITMCall
            delta = deltaCall
        else:
            price = adjK * (1 - pITMCall) - adjS * (1 - deltaCall)
            delta = deltaCall - 1

        return price, delta

    def price(
        self,
        instrument: Any,
        underlying: Stock,
        model,
        market_env: MarketEnvironment,
    ) -> float:
        """
        Compute the option price by integrating the characteristic function.

        Parameters
        ----------
        instrument : Any
            The option instrument (must have attributes 'strike', 'maturity',
            and 'option_type').
        underlying : Stock
            The underlying asset with attributes 'spot', 'volatility', and 'dividend'.
        model : BlackScholesMerton
            The option pricing model (providing a characteristic function).
        market_env : MarketEnvironment
            The market environment (providing the risk-free rate).

        Returns
        -------
        float
            The computed option price.
        """
        S = underlying.spot
        K = instrument.strike
        T = instrument.maturity
        r = market_env.rate
        q = underlying.dividend
        call_flag = instrument.option_type == "Call"

        # Obtain the characteristic function for the option.
        phi = model.characteristic_function(T, S, r, q)
        price_val, _ = self.integrate_phi(phi, S, K, r, T, q, call=call_flag)
        return price_val

    def delta(
        self,
        instrument: Any,
        underlying: Stock,
        model,
        market_env: MarketEnvironment,
    ) -> float:
        """
        Compute the option delta via integration.

        Returns
        -------
        float
            The option delta.
        """
        S = underlying.spot
        K = instrument.strike
        T = instrument.maturity
        r = market_env.rate
        q = underlying.dividend
        call_flag = instrument.option_type == "Call"

        phi = model.characteristic_function(T, S, r, q)
        _, delta_val = self.integrate_phi(phi, S, K, r, T, q, call=call_flag)
        return delta_val

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(cache_results={self._cache_results})"

    def implied_volatility(
        self,
        instrument: Any,
        underlying: Any,
        model: Any,
        market_env: Any,
        target_price: float,
        tol: float = 1e-7,
        max_iter: int = 100,
        **kwargs,
    ) -> float:
        """
        Solve for the implied volatility by reinitializing the model with each
        volatility guess. This is necessary for Fourier-based techniques where
        the characteristic function is fixed in the model instance.

        Parameters
        ----------
        instrument : Any
            The option instrument.
        underlying : Any
            The underlying asset.
        model : Any
            The pricing model. The model must provide a method `with_volatility(sigma)`
            that returns a new model instance with the updated volatility.
        market_env : Any
            The market environment.
        target_price : float
            The observed option price to match.
        tol : float, optional
            Convergence tolerance (default is 1e-7).
        max_iter : int, optional
            Maximum iterations for the root-finder (default is 100).
        **kwargs : dict
            Additional parameters.

        Returns
        -------
        float
            The implied volatility.

        Raises
        ------
        NotImplementedError
            If the model does not implement a `with_volatility` method.
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
        implied_vol = brentq(price_diff, low_vol, high_vol, xtol=tol, maxiter=max_iter)
        return implied_vol

    def vega(
        self,
        instrument: Any,
        underlying: Any,
        model: Any,
        market_env: Any,
        step: float = None,
        **kwargs,
    ) -> float:
        """
        Compute Vega (sensitivity to volatility) via finite differences with respect
        to volatility.

        For Fourier-based techniques the model's characteristic function is fixed once
        the model is instantiated. Therefore, instead of modifying the volatility in-place,
        a new model instance is created for each volatility perturbation using the model's
        `with_volatility(sigma)` method.

        Parameters
        ----------
        instrument : Any
            The option instrument.
        underlying : Any
            The underlying asset. Expected to have a 'volatility' attribute.
        model : Any
            The pricing model instance that uses volatility. The model must provide a method
            `with_volatility(sigma)` returning a new instance with the updated volatility.
        market_env : Any
            The market environment.
        step : float, optional
            The finite difference step size. If None, defaults to 1e-2 * sigma.
        **kwargs : dict
            Additional keyword arguments passed to the pricing function.

        Returns
        -------
        float
            The estimated Vega.

        Raises
        ------
        NotImplementedError
            If the model does not implement a `with_volatility` method.
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
