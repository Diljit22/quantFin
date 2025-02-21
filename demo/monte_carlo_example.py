# Suppose we want to price an American call under the Heston model

from src.stochastic.sde.sde_models import Heston
from src.techniques.monte_carlo import (
    price_american_lsm,
    call_payoff,
    rational_bounds_check,
)


def price_heston():
    # Define the Heston model
    model = Heston(
        r=0.05,
        q=0.02,
        S0=100.0,
        v0=0.04,
        kappa=1.5,
        theta=0.04,
        sigma_v=0.3,
        rho=-0.5,
        random_state=42,
    )
    T = 1.0
    n_sims = 20000
    n_steps = 50

    raw_price = price_american_lsm(
        model, lambda s: call_payoff(s, K=100.0), T, n_sims, n_steps
    )
    final_price = rational_bounds_check(
        raw_price, model.S0, 100.0, model.r, model.q, T, is_call=True
    )
    print("Heston American Call Price:", final_price)


"""
bsm_mc_examples.py

Demonstrates Monte Carlo pricing of European Call and Put options under
the Black–Scholes–Merton model for multiple strike/maturity sets.

Parameters:
-----------
Spot (S0)        = 100.0
Volatility       = 0.20
Dividend (q)     = 0.02
Risk-free (r)    = 0.05

Option sets:
1) strike=90.0, maturity=1.0
2) strike=100.0, maturity=0.75
3) strike=110.0, maturity=0.5
"""

import numpy as np


def generate_bsm_paths(
    S0: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    n_sims: int,
    n_steps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate Monte Carlo paths for a 1D Black–Scholes–Merton (GBM) process:
        dS = S*(r - q)*dt + S*sigma*dW.

    Returns an array of shape (n_sims, n_steps+1).

    Parameters
    ----------
    S0 : float
        Initial spot price
    r : float
        Risk-free rate
    q : float
        Continuous dividend yield
    sigma : float
        Volatility
    T : float
        Total time horizon (years)
    n_sims : int
        Number of paths
    n_steps : int
        Discrete time steps
    rng : np.random.Generator
        NumPy random generator (for reproducible draws)
    """
    dt = T / n_steps
    paths = np.zeros((n_sims, n_steps + 1), dtype=np.float64)
    paths[:, 0] = S0

    # Precompute drift and diffusion terms for each small dt
    drift = (r - q - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)

    for step in range(n_steps):
        Z = rng.normal(loc=0.0, scale=1.0, size=n_sims)
        # log-euler
        log_s_prev = np.log(paths[:, step])
        log_s_next = log_s_prev + drift + vol * Z
        paths[:, step + 1] = np.exp(log_s_next)

    return paths


def payoff_call(strikes: float, spots: np.ndarray) -> np.ndarray:
    """
    Vectorized call payoff: max(spots - strike, 0).
    """
    return np.maximum(spots - strikes, 0.0)


def payoff_put(strikes: float, spots: np.ndarray) -> np.ndarray:
    """
    Vectorized put payoff: max(strike - spots, 0).
    """
    return np.maximum(strikes - spots, 0.0)


def mc_price_european(
    S0: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    strike: float,
    payoff_fn,
    n_sims: int = 100_000,
    n_steps: int = 50,
    rng_seed: int = 42,
) -> float:
    """
    Monte Carlo price for a European payoff under BSM. Simple approach:
        1) generate paths
        2) discount average payoff
    """
    rng = np.random.default_rng(rng_seed)
    paths = generate_bsm_paths(S0, r, q, sigma, T, n_sims, n_steps, rng)
    ST = paths[:, -1]  # final column
    payoffs = payoff_fn(strike, ST)
    disc_factor = np.exp(-r * T)
    return disc_factor * payoffs.mean()


def main():
    spot_price = 100.0
    volatility = 0.20
    dividend = 0.02
    risk_free_rate = 0.05

    # Option sets: each param is {strike, maturity}
    params = [
        {"strike": 90.0, "maturity": 1.0},
        {"strike": 100.0, "maturity": 0.75},
        {"strike": 110.0, "maturity": 0.5},
    ]

    # We'll do both CALL and PUT for each set
    n_sims = 50_000
    n_steps = 50
    rng_seed = 42

    for param in params:
        K = param["strike"]
        T = param["maturity"]

        call_price = mc_price_european(
            S0=spot_price,
            r=risk_free_rate,
            q=dividend,
            sigma=volatility,
            T=T,
            strike=K,
            payoff_fn=payoff_call,
            n_sims=n_sims,
            n_steps=n_steps,
            rng_seed=rng_seed,
        )

        put_price = mc_price_european(
            S0=spot_price,
            r=risk_free_rate,
            q=dividend,
            sigma=volatility,
            T=T,
            strike=K,
            payoff_fn=payoff_put,
            n_sims=n_sims,
            n_steps=n_steps,
            rng_seed=rng_seed,
        )

        print(f"Option parameters: Strike={K:.1f}, Maturity={T:.2f}")
        print(f"  -> Call Price MC = {call_price:.4f}")
        print(f"  -> Put  Price MC = {put_price:.4f}")
        print("-------------------------------------------------------")


if __name__ == "__main__":
    main()
