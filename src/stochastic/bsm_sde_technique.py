"""
mc_pathwise_and_iv.py

1) Demonstrates a pathwise method for Delta (European call) in a single MC pass,
   avoiding finite-difference re-runs.

2) Illustrates a root-finding approach to implied volatility for a European call.

This code is an example. You can adapt the same pattern to:
  - More Greeks (Gamma, Vega, Rho, etc.),
  - Other SDE models (Merton jumps, Heston, etc.) by properly computing partial derivatives.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Callable


#######################################################################
# 1) Extended Base Class: sample_paths_and_derivative
#######################################################################
class BaseSDEModelExtended(ABC):
    """
    Abstract base for 1D SDE models, with:
     - sample_paths(T, n_sims, n_steps) -> array (n_sims, n_steps+1)
     - sample_paths_and_derivative(...) -> (price_array, derivative_array)

    derivative_array[i, j] = d( S_{j}^{(i)} ) / d( S_0 ), i.e. partial w.r.t. initial spot.

    Subclasses must implement these two methods in a vectorized way.
    """

    def __init__(
        self, r: float, q: float, S0: float, random_state: Optional[int] = None
    ):
        if S0 <= 0:
            raise ValueError("Initial spot S0 must be > 0.")
        self.r = r
        self.q = q
        self.S0 = S0
        self._rng = np.random.default_rng(random_state)

    @abstractmethod
    def sample_paths(self, T: float, n_sims: int, n_steps: int) -> np.ndarray:
        """
        Returns shape (n_sims, n_steps+1).
        Each row is a path from t=0..T in n_steps increments.
        """
        pass

    @abstractmethod
    def sample_paths_and_derivative(
        self, T: float, n_sims: int, n_steps: int
    ) -> (np.ndarray, np.ndarray):
        """
        Returns (S, dSdS0):
          S   shape (n_sims, n_steps+1)
          dSdS0 shape (n_sims, n_steps+1),
                partial derivative of S_{t_j} w.r.t initial spot S0.
        """
        pass

    def __repr__(self) -> str:
        cname = self.__class__.__name__
        return f"{cname}(r={self.r}, q={self.q}, S0={self.S0})"


#######################################################################
# 2) Example: BlackScholesMerton with pathwise derivative
#######################################################################
class BlackScholesMertonExtended(BaseSDEModelExtended):
    """
    Standard GBM: dS = S*(r - q)*dt + S*sigma*dW.

    We'll do a log-Euler approach for sample_paths, and track derivative
    dS_{t_{k+1}} / dS_{t_k} so we can chain back to S0.

    For each step:
      S_{k+1} = S_k * exp( (r - q - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z )
      => partial(S_{k+1}, S_k) = S_{k+1} / S_k
    => partial(S_{k+1}, S0) = partial(S_{k+1}, S_k) * partial(S_k, S0).
    """

    def __init__(
        self,
        r: float,
        q: float,
        sigma: float,
        S0: float,
        random_state: Optional[int] = None,
    ):
        super().__init__(r, q, S0, random_state)
        if sigma <= 0:
            raise ValueError("Volatility must be > 0.")
        self.sigma = sigma

    def sample_paths(self, T: float, n_sims: int, n_steps: int) -> np.ndarray:
        dt = T / n_steps
        S = np.zeros((n_sims, n_steps + 1), dtype=float)
        S[:, 0] = self.S0

        drift = (self.r - self.q - 0.5 * self.sigma**2) * dt
        vol = self.sigma * np.sqrt(dt)

        for step in range(n_steps):
            Z = self._rng.normal(0.0, 1.0, size=n_sims)
            log_s_k = np.log(S[:, step])
            log_s_next = log_s_k + drift + vol * Z
            S[:, step + 1] = np.exp(log_s_next)

        return S

    def sample_paths_and_derivative(
        self, T: float, n_sims: int, n_steps: int
    ) -> (np.ndarray, np.ndarray):
        dt = T / n_steps
        S = np.zeros((n_sims, n_steps + 1), dtype=float)
        dSdS0 = np.zeros((n_sims, n_steps + 1), dtype=float)

        # initial
        S[:, 0] = self.S0
        dSdS0[:, 0] = 1.0  # partial(S0, S0) = 1

        drift = (self.r - self.q - 0.5 * self.sigma**2) * dt
        vol = self.sigma * np.sqrt(dt)

        for step in range(n_steps):
            Z = self._rng.normal(0.0, 1.0, size=n_sims)
            log_s_k = np.log(S[:, step])
            log_s_next = log_s_k + drift + vol * Z
            S_next = np.exp(log_s_next)
            # partial wrt S_k => S_next / S_k
            # => partial(S_{k+1}, S0) = partial(S_{k+1}, S_k)*partial(S_k, S0)
            ratio = S_next / S[:, step]
            # handle S=0 edge if it arises:
            ratio = np.clip(ratio, 0.0, 1e9)

            S[:, step + 1] = S_next
            dSdS0[:, step + 1] = ratio * dSdS0[:, step]

        return S, dSdS0

    def __repr__(self) -> str:
        base = super().__repr__()
        return f"{base}, sigma={self.sigma}"


#######################################################################
# 3) Pathwise Delta for a European Call
#######################################################################
def pathwise_delta_eurocall(
    model: BaseSDEModelExtended, strike: float, T: float, n_sims: int, n_steps: int
) -> float:
    """
    Pathwise estimator of Delta for a European call payoff = max(S_T - K, 0).
    Delta = e^{-rT} * E[ 1_{S_T>K} * dS_T/dS0 ].

    Returns float scalar Delta estimate.
    """
    S, dSdS0 = model.sample_paths_and_derivative(T, n_sims, n_steps)
    ST = S[:, -1]
    dSTdS0 = dSdS0[:, -1]

    in_the_money = ST > strike
    payoff_deriv = in_the_money * dSTdS0

    disc_factor = np.exp(-model.r * T)
    delta_est = disc_factor * payoff_deriv.mean()
    return delta_est


#######################################################################
# 4) Implied Volatility for a European Call
#######################################################################
def mc_price_eurocall(
    model: BaseSDEModelExtended, strike: float, T: float, n_sims: int, n_steps: int
) -> float:
    """
    Simple MC price for a European call payoff, ignoring Greeks.
    Price = e^{-rT} * E[(S_T - K)+].
    """
    S = model.sample_paths(T, n_sims, n_steps)
    ST = S[:, -1]
    payoff = np.maximum(ST - strike, 0.0)
    disc = np.exp(-model.r * T)
    return disc * payoff.mean()


def implied_vol_mc(
    spot: float,
    r: float,
    q: float,
    T: float,
    strike: float,
    market_price: float,
    model_class: Callable[..., BaseSDEModelExtended],
    n_sims: int = 50_000,
    n_steps: int = 50,
    tol: float = 1e-4,
    max_iter: int = 50,
    seed: int = 123,
) -> float:
    """
    Finds implied volatility by bisection on MC price vs. market_price
    for a European call payoff.

    model_class: e.g. BlackScholesMertonExtended
    """
    low_vol, high_vol = 1e-6, 3.0  # bracket
    for _ in range(max_iter):
        mid_vol = 0.5 * (low_vol + high_vol)
        # Build the model with 'mid_vol'
        model = model_class(r=r, q=q, sigma=mid_vol, S0=spot, random_state=seed)
        price_mid = mc_price_eurocall(model, strike, T, n_sims, n_steps)

        diff = price_mid - market_price
        if abs(diff) < tol:
            return mid_vol

        if diff > 0:
            # MC price too high => reduce vol
            high_vol = mid_vol
        else:
            # MC price too low => increase vol
            low_vol = mid_vol

    return 0.5 * (low_vol + high_vol)


#######################################################################
# 5) Example Usage
#######################################################################
def main():
    # Example: BSM parameters
    S0 = 100.0
    r = 0.05
    q = 0.02
    sigma = 0.20
    strike = 100.0
    T = 1.0

    # Build extended BSM model
    model = BlackScholesMertonExtended(r=r, q=q, sigma=sigma, S0=S0, random_state=42)

    # 1) Pathwise Delta
    n_sims, n_steps = 100_000, 50
    delta_est = pathwise_delta_eurocall(model, strike, T, n_sims, n_steps)
    print(f"[BSM Pathwise Delta] = {delta_est:.6f} (European Call)")

    # 2) Implied Vol from a "market price" (pretend we got it from somewhere)
    market_price = 10.5
    iv_est = implied_vol_mc(
        spot=S0,
        r=r,
        q=q,
        T=T,
        strike=strike,
        market_price=market_price,
        model_class=BlackScholesMertonExtended,
        n_sims=50_000,
        n_steps=50,
        seed=123,
    )
    print(f"[Implied Vol] for market_price={market_price:.4f} => {iv_est:.4%}")


if __name__ == "__main__":
    main()
