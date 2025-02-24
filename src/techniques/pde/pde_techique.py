#!/usr/bin/env python3
"""
pde_technique.py
================

PDETechnique using a Crank–Nicolson finite-difference scheme for option pricing.
This technique is implemented as a subclass of FiniteDifferenceTechnique, so it inherits
finite-difference-based Greek calculations.

Usage Example:
    from pde_technique import PDETechnique
    from src.models.black_scholes_merton import BlackScholesMerton
    from src.market.market_environment import MarketEnvironment
    from src.instruments.base_option import BaseOption as Instrument
    from src.underlyings.stock import Stock as Underlying

    underlying = Underlying(spot=100.0, volatility=0.2, dividend=0.02)
    market_env = MarketEnvironment(rate=0.05)
    model = BlackScholesMerton(sigma=0.2)
    pde_solver = PDETechnique(S_max=600.0, M=256, N=256, parallel=True)
    
    price = pde_solver.price(instrument, underlying, model, market_env)
    print("Option Price:", price)
"""

import math
import numpy as np
from scipy.linalg import solve_banded
from typing import Any
import copy

# Import the base finite-difference technique that includes GreekMixin functionality.
from src.techniques.finite_diff_technique import FiniteDifferenceTechnique
from src.models.base_model import BaseModel
from src.market.market_environment import MarketEnvironment
from src.instruments.base_option import BaseOption as Instrument
from src.underlyings.stock import Stock as Underlying


class PDETechnique(FiniteDifferenceTechnique):
    def __init__(
        self,
        S_max: float = 300.0,
        M: int = 200,
        N: int = 200,
        scheme: str = "CrankNicolson",
        concurrency: bool = False,
        cache_results: bool = False,
        parallel: bool = False,
    ) -> None:
        """
        Initialize the PDETechnique.

        Parameters
        ----------
        S_max : float
            Maximum asset price in the grid.
        M : int
            Number of spatial steps (grid points = M+1).
        N : int
            Number of time steps.
        scheme : str
            Finite-difference scheme (default: "CrankNicolson").
        concurrency : bool
            Flag to enable parallel computations.
        cache_results : bool
            Whether to cache price and IV results.
        parallel : bool
            If True, enables parallel finite-difference computations for Greeks.
        """
        # Initialize the base FiniteDifferenceTechnique, which sets up GreekMixin attributes.
        super().__init__(cache_results=cache_results, parallel=parallel)
        self.S_max = S_max
        self.M = M
        self.N = N
        self.scheme = scheme.lower()
        self.concurrency = concurrency
        # Explicitly set _parallel_fd so that GreekMixin can reference it.
        self._parallel_fd = parallel

    def price(
        self,
        instrument: Instrument,
        underlying: Underlying,
        model: BaseModel,
        market_env: MarketEnvironment,
        **kwargs,
    ) -> float:
        """
        Solve the PDE to price the option.

        Parameters
        ----------
        instrument : Instrument
            The option instrument (must have strike, maturity, and option_type).
        underlying : Underlying
            The underlying asset (must have spot, volatility, and dividend).
        model : BaseModel
            The pricing model providing the PDE coefficients.
        market_env : MarketEnvironment
            The market environment providing the risk-free rate.

        Returns
        -------
        float
            The computed option price at S = underlying.spot at t = 0.
        """
        # Extract parameters.
        r = market_env.rate
        q = underlying.dividend
        sigma = underlying.volatility
        K = instrument.strike
        T = instrument.maturity
        S0 = underlying.spot
        option_type = instrument.option_type.lower()

        M = self.M
        S_max = self.S_max
        dS = S_max / M
        S_grid = np.linspace(0, S_max, M + 1)
        N = self.N
        dt = T / N

        # Terminal condition: option payoff at maturity.
        if option_type == "call":
            V = np.maximum(S_grid - K, 0)
        else:
            V = np.maximum(K - S_grid, 0)

        # Precompute coefficients for the interior grid (M-1 points).
        M_int = M - 1
        a = np.zeros(M_int)
        b = np.zeros(M_int)
        c = np.zeros(M_int)
        aR = np.zeros(M_int)
        bR = np.zeros(M_int)
        cR = np.zeros(M_int)

        for i in range(1, M):
            S_val = S_grid[i]
            A_i, B_i, C_i = model.pde(S_val, r, q, K, T)
            a[i - 1] = -(dt / 2) * (A_i / dS**2 - B_i / (2 * dS))
            b[i - 1] = 1 + (dt / 2) * (2 * A_i / dS**2 + C_i)
            c[i - 1] = -(dt / 2) * (A_i / dS**2 + B_i / (2 * dS))
            aR[i - 1] = (dt / 2) * (A_i / dS**2 - B_i / (2 * dS))
            bR[i - 1] = 1 - (dt / 2) * (2 * A_i / dS**2 + C_i)
            cR[i - 1] = (dt / 2) * (A_i / dS**2 + B_i / (2 * dS))

        # Assemble the banded matrix for the left-hand side.
        ab = np.zeros((3, M_int))
        ab[0, 1:] = c[:-1]  # upper diagonal.
        ab[1, :] = b  # main diagonal.
        ab[2, :-1] = a[1:]  # lower diagonal.

        # Time stepping: backward in time from T to 0.
        for n in range(N):
            t = T - n * dt
            if option_type == "call":
                V0 = 0.0
                V_M = S_max - K * math.exp(-r * (T - t))
            else:
                V0 = K * math.exp(-r * (T - t))
                V_M = 0.0

            # Build RHS for the interior nodes.
            rhs = np.zeros(M_int)
            for i in range(1, M):
                idx = i - 1
                rhs[idx] = aR[idx] * V[i - 1] + bR[idx] * V[i] + cR[idx] * V[i + 1]
            rhs[0] -= a[0] * V0
            rhs[-1] -= c[-1] * V_M

            # Solve the tridiagonal system.
            V_new = solve_banded((1, 1), ab, rhs)
            V[1:M] = V_new
            V[0] = V0
            V[M] = V_M

        # Interpolate to get the price at S0.
        if S0 <= 0:
            return V[0]
        elif S0 >= S_max:
            return V[-1]
        else:
            j = np.searchsorted(S_grid, S0) - 1
            w = (S0 - S_grid[j]) / dS
            return V[j] * (1 - w) + V[j + 1] * w

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(S_max={self.S_max}, M={self.M}, N={self.N}, "
            f"scheme='{self.scheme}', concurrency={self.concurrency})"
        )

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

    def gamma(
        self,
        instrument: Any,
        underlying: Any,
        model: Any,
        market_env: Any,
        step: float = None,
        **kwargs,
    ) -> float:
        """
        Compute Gamma (the second derivative of the option price with respect to the underlying spot)
        using a five-point finite-difference approximation.

        Parameters
        ----------
        instrument : Any
            The option instrument.
        underlying : Any
            The underlying asset (expected to have a 'spot' attribute).
        model : Any
            The pricing model instance.
        market_env : Any
            The market environment.
        step : float, optional
            The finite difference step size. If None, defaults to 1e-2 * underlying.spot.
        **kwargs : dict
            Additional parameters passed to the pricing function.

        Returns
        -------
        float
            The estimated Gamma.
        """
        # Get the current spot price.
        S0 = underlying.spot
        if step is None:
            step = 1e-2 * S0 if S0 != 0 else 1e-2

        # Create new Underlying instances at S0 ± h and S0 ± 2h.
        underlying_p2 = Underlying(
            S0 + 2 * step, underlying.volatility, underlying.dividend
        )
        underlying_p1 = Underlying(
            S0 + step, underlying.volatility, underlying.dividend
        )
        underlying_m1 = Underlying(
            S0 - step, underlying.volatility, underlying.dividend
        )
        underlying_m2 = Underlying(
            S0 - 2 * step, underlying.volatility, underlying.dividend
        )

        # Compute prices at S0, S0 + h, S0 - h, S0 + 2h, and S0 - 2h.
        f0 = self.price(instrument, underlying, model, market_env, **kwargs)
        f1 = self.price(instrument, underlying_p1, model, market_env, **kwargs)
        f_1 = self.price(instrument, underlying_m1, model, market_env, **kwargs)
        f2 = self.price(instrument, underlying_p2, model, market_env, **kwargs)
        f_2 = self.price(instrument, underlying_m2, model, market_env, **kwargs)

        # Apply the five-point stencil formula.
        gamma_val = (-f2 + 16 * f1 - 30 * f0 + 16 * f_1 - f_2) / (4 * step**2)
        return gamma_val

    def gamma2(
        self,
        instrument: Any,
        underlying: Any,
        model: Any,
        market_env: Any,
        step: float = None,
        **kwargs,
    ) -> float:
        """
        Compute Gamma (the second derivative of the option price with respect to the underlying spot)
        via a central finite-difference approximation.

        Parameters
        ----------
        instrument : Any
            The option instrument.
        underlying : Any
            The underlying asset (expected to have a 'spot' attribute).
        model : Any
            The pricing model instance.
        market_env : Any
            The market environment.
        step : float, optional
            The finite difference step size. If None, defaults to 1e-2 * underlying.spot.
        **kwargs : dict
            Additional parameters passed to the pricing function.

        Returns
        -------
        float
            The estimated Gamma.
        """
        # Get the current spot price.
        S0 = underlying.spot
        if step is None:
            step = 1e-2 * S0 if S0 != 0 else 1e-2

        # Instead of deepcopy, create new Underlying instances using the same parameters.
        underlying_up = Underlying(
            S0 + step, underlying.volatility, underlying.dividend
        )
        underlying_down = Underlying(
            S0 - step, underlying.volatility, underlying.dividend
        )

        # Compute prices at S0, S0+step, and S0-step.
        price_center = self.price(instrument, underlying, model, market_env, **kwargs)
        price_up = self.price(instrument, underlying_up, model, market_env, **kwargs)
        price_down = self.price(
            instrument, underlying_down, model, market_env, **kwargs
        )

        # Use the central finite-difference formula for the second derivative.
        gamma_val = 2 * (price_up - 2 * price_center + price_down) / (step**2)
        return gamma_val
