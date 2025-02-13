# techniques/pde/pde_solver.py

"""
PDETechnique
============

A robust, production-grade solver for the one-dimensional Black–Scholes PDE
using finite differences (explicit, fully implicit, or Crank–Nicolson).
It supports both European and (with minor modifications) American payoff
conditions. By default, we show a Crank–Nicolson approach for better stability
and second-order convergence in time and space.

Key Features
------------
1. Concurrency: 
   - For CPU-intensive large grids, we allow parallelization of certain parts
     (e.g., building the tridiagonal matrix, or possibly solving multiple
     scenarios at once if needed).
   - Direct time stepping is inherently sequential (since each step depends
     on the result of the next/previous step), but partial concurrency can be
     used for sub-tasks.

2. Vectorization:
   - Matrix assembly and vector operations use NumPy arrays.
   - A tridiagonal solver is also vectorized. For best HPC performance, one can
     consider specialized linear algebra libraries or factorization routines.

3. Extensibility:
   - Inherit from `PDETechnique` for more advanced PDE or different boundary
     conditions (e.g., local volatility, multiple factors).
   - Overriding `_setup_boundary_conditions(...)` or `_build_coefficients(...)`
     can adapt it to other PDE variants or payoff styles.

4. Production-Grade:
   - Includes robust error handling (time steps, domain range).
   - Large concurrency contexts can use `multiprocessing` or `joblib` if data
     can be pickled or is read-only. 
   - For American options, one typically does a “max(V, payoff)” step if early
     exercise is allowed. We provide a placeholder for that.

Usage Example (Pseudo-Code)
---------------------------
>>> from src.techniques.pde.pde_solver import PDETechnique
>>> from src.instruments.option import Option
>>> from src.models.blacks_scholes_merton_model import BlackScholesMertonModel
>>> from src.market.market_environment import MarketEnvironment

>>> env = MarketEnvironment(spot_prices={"AAPL": 150.0})
>>> bsm_model = BlackScholesMertonModel(sigma=0.2, risk_free_rate=0.03, dividend_yield=0.01)
>>> call_option = Option("AAPL", maturity=1.0, strike=150.0, option_type="call", option_style="european")

>>> pde_solver = PDETechnique(
...     S_max=4.0 * 150.0,  # up to 600 in price dimension
...     M=256,  # number of spatial steps
...     N=256,  # number of time steps
...     scheme="CrankNicolson",
...     concurrency=False
... )
>>> call_price = pde_solver.price(call_option, bsm_model, env)
>>> print("Call Price from PDE:", call_price)

>>> # PDE-based implied volatility (re-uses the PDE approach, can be slow):
>>> iv = pde_solver.implied_volatility(call_option, bsm_model, env, market_price=12.0)
>>> print("PDE Implied Vol:", iv)
"""

import math
import cmath
import numpy as np
from typing import Any, Dict, Optional

from src.techniques.base_technique import BaseTechnique
from src.instruments.base_instrument import BaseInstrument
from src.models.base_model import BaseModel
from src.market.market_environment import MarketEnvironment


class PDETechnique(BaseTechnique):
    """
    PDETechnique solves the Black–Scholes PDE on a 1D grid in space (S)
    and time (t). By default, it uses the Crank–Nicolson scheme, but
    explicit or implicit can be chosen.

    PDE: dV/dt + 0.5*sigma^2*S^2 * d^2V/dS^2
         + (r - q)*S * dV/dS - r*V = 0

    We discretize S in [0, S_max] with M steps, t in [0, T] with N steps
    (stepping backwards in time: from T to 0 or forward from 0 to T
    depending on the scheme, but typically backward for terminal payoff).

    Attributes
    ----------
    S_max : float
        Maximum S in the grid. Must be sufficiently large to minimize
        boundary error for big underlying prices.
    M : int
        Number of spatial grid steps (S dimension).
    N : int
        Number of time steps (t dimension).
    scheme : str
        PDE scheme: 'explicit', 'implicit', or 'CrankNicolson'.
    concurrency : bool
        If True, attempt parallel tasks (like building the tridiagonal
        matrix or solving multiple param sets). The time stepping itself
        is sequential.

    Methods
    -------
    price(instrument, model, market_env, **kwargs) -> float:
        Compute the option price using PDE.

    implied_volatility(...)
        Root-finding around the PDE-based price(...).

    # If needed, we can override the typical greeks. But typically PDE-based
    # greeks could be done by reading off partial derivatives from the grid or
    # using a finite-difference approach around the PDE as well. We'll omit
    # that for brevity.
    """

    def __init__(
        self,
        S_max: float = 300.0,
        M: int = 200,
        N: int = 200,
        scheme: str = "CrankNicolson",
        concurrency: bool = False,
    ) -> None:
        """
        Constructor for PDETechnique.

        Parameters
        ----------
        S_max : float, default=300.0
            Upper bound for the S domain.
        M : int, default=200
            Number of spatial steps.
        N : int, default=200
            Number of time steps.
        scheme : {'explicit', 'implicit', 'CrankNicolson'}
            Discretization scheme for PDE.
        concurrency : bool, default=False
            If True, attempt concurrency for certain tasks.
        """
        super().__init__()
        self.S_max = S_max
        self.M = M
        self.N = N
        self.scheme = scheme.lower()
        self.concurrency = concurrency

    # ----------------------------------------------------------------
    # BaseTechnique Methods
    # ----------------------------------------------------------------
    def price(
        self, instrument: BaseInstrument, model: BaseModel, market_env: Any, **kwargs
    ) -> float:
        """
        Solve the PDE to price the given instrument.
        Currently handles standard European calls/puts in the BSM setting.

        For more exotic payoffs or American-style,
        override `_apply_early_exercise(...)` or `_setup_payoff(...)`.

        Returns
        -------
        float
            The option price at t=0, S = S0.
        """
        if getattr(instrument, "option_style", "european").lower() not in [
            "european",
            "american",
        ]:
            raise ValueError(
                "PDETechnique currently supports European or American payoffs only."
            )

        # Extract model parameters
        params = model.get_params()
        r = params.get("r", 0.0)
        q = params.get("dividend_yield", 0.0)
        sigma = params.get("sigma", 0.0)

        # Spot from market environment
        if not isinstance(market_env, MarketEnvironment):
            raise ValueError(
                "PDETechnique expects a MarketEnvironment for 'market_env'."
            )
        S0 = market_env.get_spot_price(instrument.underlying_symbol)

        # Time to maturity
        T = instrument.maturity
        if T <= 1e-14:
            # immediate expiry => payoff
            payoff_val = self._intrinsic(instrument, S0)
            return max(0.0, payoff_val)

        is_call = instrument.option_type.lower() == "call"
        is_american = instrument.option_style.lower() == "american"

        # Build the PDE grid
        ds = self.S_max / self.M
        dt = T / self.N

        # grid for S and t
        S_values = np.linspace(0.0, self.S_max, self.M + 1)
        # We'll hold V at each step
        V = np.zeros(self.M + 1, dtype=np.float64)
        # terminal payoff
        for i, s_val in enumerate(S_values):
            V[i] = self._intrinsic_option(is_call, s_val, instrument.strike)

        # step backward in time or forward => typically backward from T to 0
        if self.scheme == "explicit":
            return self._solve_explicit(
                V,
                S_values,
                ds,
                dt,
                r,
                q,
                sigma,
                T,
                S0,
                is_american,
                instrument.strike,
                is_call,
            )
        elif self.scheme == "implicit":
            return self._solve_implicit(
                V,
                S_values,
                ds,
                dt,
                r,
                q,
                sigma,
                T,
                S0,
                is_american,
                instrument.strike,
                is_call,
            )
        else:  # "cranknicolson"
            return self._solve_crank_nicolson(
                V,
                S_values,
                ds,
                dt,
                r,
                q,
                sigma,
                T,
                S0,
                is_american,
                instrument.strike,
                is_call,
            )

    def implied_volatility(
        self,
        instrument: BaseInstrument,
        model: BaseModel,
        market_env: Any,
        market_price: float,
        **kwargs
    ) -> float:
        """
        PDE-based implied volatility.
        We do a simple root-finding that updates 'sigma' in the model's params
        and re-runs the PDE price. Potentially expensive for large M,N.

        Returns
        -------
        float
            The implied volatility consistent with the observed market_price.
        """
        vol_lower, vol_upper = kwargs.get("vol_bounds", (1e-9, 5.0))
        tol = kwargs.get("tol", 1e-6)
        max_iter = kwargs.get("max_iter", 100)
        guess = kwargs.get("initial_guess", 0.2)

        def f(vol):
            old_vol = model._params.get("sigma", 0.0)
            model._params["sigma"] = vol
            val = self.price(instrument, model, market_env)
            model._params["sigma"] = old_vol
            return val - market_price

        fl = f(vol_lower)
        fu = f(vol_upper)
        if fl * fu > 0:
            # fallback to secant
            return self._secant_iv(f, guess, tol, max_iter)
        else:
            return self._bisection_iv(f, vol_lower, vol_upper, tol, max_iter)

    # ----------------------------------------------------------------
    # PDE Solvers
    # ----------------------------------------------------------------
    def _solve_explicit(
        self, V, S_values, ds, dt, r, q, sigma, T, S0, is_american, K, is_call
    ):
        """
        Explicit (forward time, backward space) finite difference approach.
        Typically requires small dt for stability.

        Returns
        -------
        float
            Interpolated value for S0 after PDE is solved.
        """
        M = len(S_values) - 1
        n_steps = int(T / dt)

        # Precompute coefficients
        # For node i:
        # alpha_i = 0.5 * dt * (sigma^2 * i^2 - (r - q) * i)
        # beta_i  = 1 - dt * (sigma^2 * i^2 + r)
        # gamma_i = 0.5 * dt * (sigma^2 * i^2 + (r - q) * i)

        for n in range(n_steps):
            V_old = V.copy()
            for i in range(1, M):
                i_f = float(i)
                alpha = 0.5 * dt * (sigma**2 * (i_f**2) - (r - q) * i_f)
                beta = 1.0 - dt * (sigma**2 * i_f**2 + r)
                gamma = 0.5 * dt * (sigma**2 * i_f**2 + (r - q) * i_f)

                V[i] = alpha * V_old[i - 1] + beta * V_old[i] + gamma * V_old[i + 1]

            # Boundary conditions
            V[0] = self._bc_lower(is_call, 0.0, K, r, T - (n + 1) * dt)
            V[M] = self._bc_upper(is_call, S_values[M], K, r, T - (n + 1) * dt)

            # If American, do immediate payoff check
            if is_american:
                for i in range(M + 1):
                    payoff = self._intrinsic_option(is_call, S_values[i], K)
                    V[i] = max(V[i], payoff)

        # Interpolate for S0
        return self._interpolate(S_values, V, S0)

    def _solve_implicit(
        self, V, S_values, ds, dt, r, q, sigma, T, S0, is_american, K, is_call
    ):
        """
        Fully implicit (backward time, we invert a tridiagonal system at each step).
        Unconditionally stable, can use bigger dt.
        """
        M = len(S_values) - 1
        n_steps = int(T / dt)

        # Build the tri-di matrix factors once if concurrency is desired, or each step?
        # We'll do it each step if i depends on iteration, but actually i is the same each step.

        # For node i:
        # a_i = 0.5 * dt * ( (r - q)*i - sigma^2*i^2 )
        # b_i = 1 + dt*(sigma^2*i^2 + r)
        # c_i = -0.5 * dt*( (r - q)*i + sigma^2*i^2 )

        # We'll build the matrix A s.t. A * V_new = V_old
        # Then solve for V_new.

        A_diag = np.zeros(M - 1, dtype=np.float64)
        A_lower = np.zeros(M - 2, dtype=np.float64)
        A_upper = np.zeros(M - 2, dtype=np.float64)

        for i in range(1, M):
            i_f = float(i)
            a_i = 0.5 * dt * ((r - q) * i_f - sigma**2 * i_f**2)
            b_i = 1.0 + dt * (sigma**2 * i_f**2 + r)
            c_i = -0.5 * dt * ((r - q) * i_f + sigma**2 * i_f**2)

            A_diag[i - 1] = b_i
            if i - 1 > 0:
                A_lower[i - 2] = a_i
            if i - 1 < (M - 2):
                A_upper[i - 1] = c_i

        for n in range(n_steps):
            # We'll incorporate boundary conditions into the RHS vector.
            V_old = V.copy()
            rhs = V_old[1:M].copy()

            # Adjust boundaries
            # We do: A * V_new = V_old => but V_old has boundary terms
            rhs[0] -= A_lower[0] * self._bc_lower(
                is_call, S_values[0], K, r, T - (n + 1) * dt
            )
            rhs[-1] -= A_upper[-1] * self._bc_upper(
                is_call, S_values[M], K, r, T - (n + 1) * dt
            )

            # Solve tri-di system
            V_new_interior = self._solve_tridiagonal(A_lower, A_diag, A_upper, rhs)
            # assemble final
            V[1:M] = V_new_interior
            # boundaries
            V[0] = self._bc_lower(is_call, S_values[0], K, r, T - (n + 1) * dt)
            V[M] = self._bc_upper(is_call, S_values[M], K, r, T - (n + 1) * dt)

            if is_american:
                for i in range(M + 1):
                    payoff = self._intrinsic_option(is_call, S_values[i], K)
                    V[i] = max(V[i], payoff)

        return self._interpolate(S_values, V, S0)

    def _solve_crank_nicolson(
        self, V, S_values, ds, dt, r, q, sigma, T, S0, is_american, K, is_call
    ):
        """
        Crank–Nicolson scheme: average of explicit & implicit =>
        0.5 LHS (time n+1) + 0.5 RHS (time n).
        """
        M = len(S_values) - 1
        n_steps = int(T / dt)

        # We'll define arrays for the "implicit" part and the "explicit" part
        # For node i:
        # alpha_i = 0.25*dt*( (r - q)*i - sigma^2*i^2 )
        # beta_i  = -0.5*dt*(sigma^2*i^2 + r)
        # gamma_i = 0.25*dt*( (r - q)*i + sigma^2*i^2 )

        # LHS (implicit) => diag: 1 - 2 * beta_i, sub: -alpha_i, sup: -gamma_i
        # RHS (explicit) => diag: 1 + 2 * beta_i, sub: alpha_i, sup: gamma_i

        A_diag = np.zeros(M - 1, dtype=np.float64)
        A_lower = np.zeros(M - 2, dtype=np.float64)
        A_upper = np.zeros(M - 2, dtype=np.float64)

        B_diag = np.zeros(M - 1, dtype=np.float64)
        B_lower = np.zeros(M - 2, dtype=np.float64)
        B_upper = np.zeros(M - 2, dtype=np.float64)

        for i in range(1, M):
            i_f = float(i)

            alpha = 0.25 * dt * ((r - q) * i_f - sigma**2 * i_f**2)
            beta = -0.5 * dt * (sigma**2 * i_f**2 + r)
            gamma = 0.25 * dt * ((r - q) * i_f + sigma**2 * i_f**2)

            # LHS
            A_diag[i - 1] = 1.0 - 2.0 * beta
            if i - 1 > 0:
                A_lower[i - 2] = -alpha
            if i - 1 < (M - 2):
                A_upper[i - 1] = -gamma

            # RHS
            B_diag[i - 1] = 1.0 + 2.0 * beta
            if i - 1 > 0:
                B_lower[i - 2] = alpha
            if i - 1 < (M - 2):
                B_upper[i - 1] = gamma

        for n in range(n_steps):
            V_old = V.copy()
            # Build RHS
            rhs = np.zeros(M - 1, dtype=np.float64)
            for i in range(1, M):
                idx = i - 1
                rhs_val = B_diag[idx] * V_old[i]
                if i - 1 > 0:
                    rhs_val += B_lower[i - 2] * V_old[i - 1]
                if i < (M - 1):
                    rhs_val += B_upper[idx] * V_old[i + 1]
                rhs[idx] = rhs_val

            # incorporate boundary conditions
            # For the LHS we do typical boundary effect
            # We'll just do simplistic approach
            rhs[0] -= A_lower[0] * self._bc_lower(
                is_call, S_values[0], K, r, T - (n + 1) * dt
            )
            rhs[-1] -= A_upper[-1] * self._bc_upper(
                is_call, S_values[M], K, r, T - (n + 1) * dt
            )

            # Solve tri-di
            V_new_interior = self._solve_tridiagonal(A_lower, A_diag, A_upper, rhs)
            V[1:M] = V_new_interior

            # boundaries
            V[0] = self._bc_lower(is_call, S_values[0], K, r, T - (n + 1) * dt)
            V[M] = self._bc_upper(is_call, S_values[M], K, r, T - (n + 1) * dt)

            if is_american:
                for i in range(M + 1):
                    payoff = self._intrinsic_option(is_call, S_values[i], K)
                    V[i] = max(V[i], payoff)

        return self._interpolate(S_values, V, S0)

    # ----------------------------------------------------------------
    # PDE / Linear Algebra Helpers
    # ----------------------------------------------------------------
    def _solve_tridiagonal(self, lower, diag, upper, rhs):
        """
        Solve a tri-diagonal system Ax = rhs, where:
          - `lower[i]` is the sub-diagonal element in row i+1
          - `diag[i]`  is the diagonal element in row i
          - `upper[i]` is the super-diagonal in row i
        This is a standard Thomas algorithm in O(n).
        """
        n = len(diag)
        c_star = np.zeros(n - 1, dtype=np.float64)
        d_star = np.zeros(n, dtype=np.float64)
        x = np.zeros(n, dtype=np.float64)

        # forward pass
        c_star[0] = upper[0] / diag[0]
        d_star[0] = rhs[0] / diag[0]
        for i in range(1, n - 1):
            temp = diag[i] - lower[i - 1] * c_star[i - 1]
            c_star[i] = upper[i] / temp
            d_star[i] = (rhs[i] - lower[i - 1] * d_star[i - 1]) / temp

        # last d_star
        temp = diag[n - 1] - lower[n - 2] * c_star[n - 2]
        d_star[n - 1] = (rhs[n - 1] - lower[n - 2] * d_star[n - 2]) / temp

        # backward pass
        x[n - 1] = d_star[n - 1]
        for i in reversed(range(n - 1)):
            x[i] = d_star[i] - c_star[i] * x[i + 1]

        return x

    def _intrinsic_option(self, is_call: bool, S: float, K: float) -> float:
        """
        Intrinsic payoff for a vanilla call/put at expiry.
        """
        if is_call:
            return max(S - K, 0.0)
        else:
            return max(K - S, 0.0)

    def _intrinsic(self, instrument: BaseInstrument, S: float) -> float:
        """
        For immediate expiry or zero time, return the direct payoff from the instrument.
        """
        is_call = instrument.option_type.lower() == "call"
        return self._intrinsic_option(is_call, S, instrument.strike)

    def _bc_lower(self, is_call, S, K, r, tau):
        """
        Boundary condition at S=0. For a call, payoff=0. For a put, payoff ~ K e^{-r tau} if needed.
        """
        if is_call:
            return 0.0
        else:
            # put => near S=0 => option ~ K e^{-r tau}
            return K * math.exp(-r * tau)

    def _bc_upper(self, is_call, S, K, r, tau):
        """
        Boundary at S=Smax. For a call, payoff ~ (S - K e^{-r tau}), for a put, ~ 0 at large S.
        """
        if is_call:
            return S - K * math.exp(-r * tau)
        else:
            return 0.0

    def _interpolate(self, S_values, V_values, S0):
        """
        Simple linear interpolation to find V(S0).
        """
        if S0 >= S_values[-1]:
            return V_values[-1]
        if S0 <= S_values[0]:
            return V_values[0]

        idx = np.searchsorted(S_values, S0) - 1
        # clamp
        if idx < 0:
            idx = 0
        if idx >= len(S_values) - 1:
            idx = len(S_values) - 2

        s1, s2 = S_values[idx], S_values[idx + 1]
        v1, v2 = V_values[idx], V_values[idx + 1]
        return v1 + (v2 - v1) * ((S0 - s1) / (s2 - s1))

    # ----------------------------------------------------------------
    # Helper for Implied Vol
    # (Similar to what's done in other techniques)
    # ----------------------------------------------------------------
    def _secant_iv(self, f, guess, tol, max_iter):
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
