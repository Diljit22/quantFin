"""
test_bsm_mc.py

Demonstration of:
  - Using the BlackScholesMerton class from sde_models.py
  - Pricing multiple European Calls/Puts via MC (option_pricing.py)
  - Comparing the MC results to the known closed-form Black–Scholes formula.

Model Parameters:
----------------
Spot (S0)        = 100.0
Volatility       = 0.20
Dividend (q)     = 0.02
Risk-free (r)    = 0.05

Option sets: (strike, maturity)
1) (90.0, 1.0)
2) (100.0, 0.75)
3) (110.0, 0.50)

We price both a Call and a Put for each set, then compare to the
Black–Scholes closed-form formula as a reference.
"""

import math
import numpy as np

# Import from the previously defined modules:
from src.stochastic.sde.sde_models import BlackScholesMerton
from src.techniques.monte_carlo import price_european, call_payoff, put_payoff


#########################################
# 1) Analytical Black–Scholes Formulas
#########################################
def black_scholes_call(
    S0: float, K: float, r: float, q: float, sigma: float, T: float
) -> float:
    """
    Closed-form for a European Call option under Black–Scholes.
    """
    from math import log, sqrt, exp
    from scipy.stats import norm

    if T <= 0.0:
        return max(0.0, S0 - K)  # immediate payoff

    d1 = (math.log(S0 / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    # discounted underlying minus discounted strike
    return S0 * exp(-q * T) * Nd1 - K * exp(-r * T) * Nd2


def black_scholes_put(
    S0: float, K: float, r: float, q: float, sigma: float, T: float
) -> float:
    """
    Closed-form for a European Put option under Black–Scholes.
    """
    from math import log, sqrt, exp
    from scipy.stats import norm

    if T <= 0.0:
        return max(0.0, K - S0)

    d1 = (math.log(S0 / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    return K * exp(-r * T) * (1.0 - Nd2) - S0 * exp(-q * T) * (1.0 - Nd1)


#########################################
# 2) Main Example
#########################################
def main():
    # Common model parameters
    spot_price = 100.0
    volatility = 0.20
    dividend = 0.02
    risk_free_rate = 0.05

    # Option sets: (strike, maturity)
    option_params = [
        (90.0, 1.0),
        (100.0, 0.75),
        (110.0, 0.50),
    ]

    # Instantiate the BSM model from sde_models
    model = BlackScholesMerton(
        r=risk_free_rate,
        q=dividend,
        sigma=volatility,
        S0=spot_price,
        random_state=42,  # for reproducible draws
    )

    # MC settings
    n_sims = 100000
    n_steps = 100

    print("=== BSM Monte Carlo Demo ===\n")
    print(f"Model: {model}")
    print(f"MC config: n_sims={n_sims}, n_steps={n_steps}")

    for K, T in option_params:
        # (1) Price a Call via MC
        call_mc = price_european(
            model=model,
            payoff_fn=lambda paths: call_payoff(paths, K),
            T=T,
            n_sims=n_sims,
            n_steps=n_steps,
        )
        # (2) Price a Put via MC
        put_mc = price_european(
            model=model,
            payoff_fn=lambda paths: put_payoff(paths, K),
            T=T,
            n_sims=n_sims,
            n_steps=n_steps,
        )

        # (3) Compute closed-form
        call_cf = black_scholes_call(
            spot_price, K, risk_free_rate, dividend, volatility, T
        )
        put_cf = black_scholes_put(
            spot_price, K, risk_free_rate, dividend, volatility, T
        )

        # Print results
        print(f"\nStrike={K:.2f}, Maturity={T:.2f}")
        print(
            f" Call:  MC={call_mc:.4f},   BS={call_cf:.4f},  Diff={call_mc - call_cf:.4f}"
        )
        print(
            f" Put :  MC={put_mc:.4f},   BS={put_cf:.4f},  Diff={put_mc - put_cf:.4f}"
        )


if __name__ == "__main__":
    main()
