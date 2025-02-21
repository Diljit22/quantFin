"""
lr_lattice.py
=============

This module implements a Leisen–Reimer lattice for pricing European options.
The tree is constructed with an odd number of steps and is designed solely to compute
a price that converges to the Black–Scholes value.

Usage Example:
--------------
    >>> lr_call = LeisenReimerLattice(S0=100, K=100, r=0.05, sigma=0.2, q=0.0, T=1.0, steps=101, is_call=True)
    >>> print("European Call Price =", lr_call.price_option())
    >>> lr_put = LeisenReimerLattice(S0=100, K=100, r=0.05, sigma=0.2, q=0.0, T=1.0, steps=101, is_call=False)
    >>> print("European Put Price =", lr_put.price_option())
"""

import math
import numpy as np
from scipy.stats import norm


def peizer_pratt(z, n):
    """
    Inverse–Peizer–Pratt function used in the Leisen–Reimer model.

    Parameters:
        z : float
            The input (typically d₂ or d₁).
        n : int
            The number of steps in the binomial model.

    Returns:
        float : a probability between 0 and 1.

    Formula:
        h⁻¹(z; n) = 0.5 + 0.5·sign(z)·sqrt[1 – exp(–z²/(n + 1/3 – 0.1/(n+1)))]
    """
    denom = n + 1.0 / 3.0 - 0.1 / (n + 1)
    return 0.5 + 0.5 * np.sign(z) * np.sqrt(1 - np.exp(-(z * z) / denom))


class LeisenReimerLattice:
    def __init__(self, S0, K, r, sigma, q, T, steps, is_call=True):
        """
        Initialize the Leisen–Reimer lattice for European options.

        Parameters:
            S0      : float, initial stock price.
            K       : float, strike price.
            r       : float, risk–free rate.
            sigma   : float, volatility.
            q       : float, dividend yield.
            T       : float, time to maturity (in years).
            steps   : int, number of time steps (will be forced to be odd).
            is_call : bool, True for call options, False for put options.
        """
        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.q = q
        self.T = T
        # Force an odd number of steps.
        if steps % 2 == 0:
            steps += 1
        self.steps = steps
        self.dt = T / steps
        self.exp_dt = math.exp((r - q) * self.dt)

        # Compute Black–Scholes d1 and d2.
        self.d1 = (math.log(S0 / K) + (r - q + 0.5 * sigma * sigma) * T) / (
            sigma * math.sqrt(T)
        )
        self.d2 = self.d1 - sigma * math.sqrt(T)

        n = self.steps
        # Compute the risk–neutral probability (p) and auxiliary probability (p′)
        if is_call:
            if S0 >= K:
                self.p = peizer_pratt(self.d2, n)
                self.p_prime = peizer_pratt(self.d1, n)
            else:
                self.p = 1 - peizer_pratt(-self.d2, n)
                self.p_prime = 1 - peizer_pratt(-self.d1, n)
        else:
            # For put options, use put–call symmetry.
            if S0 >= K:
                self.p = 1 - peizer_pratt(-self.d1, n)
                self.p_prime = 1 - peizer_pratt(-self.d2, n)
            else:
                self.p = peizer_pratt(self.d1, n)
                self.p_prime = peizer_pratt(self.d2, n)

        # Calculate move multipliers so that the tree is centered around the strike.
        self.u = self.exp_dt * (self.p_prime / self.p)
        self.d = self.exp_dt * ((1 - self.p_prime) / (1 - self.p))
        self.is_call = is_call

    def price_option(self):
        """
        Build the recombining binomial tree and compute the European option price.

        Returns:
            float: The option price.
        """
        n = self.steps
        # Allocate arrays for stock prices and option values.
        stock = np.zeros((n + 1, n + 1))
        option = np.zeros((n + 1, n + 1))
        # Build the tree by forward induction.
        for t in range(n + 1):
            for i in range(t + 1):
                stock[t, i] = self.S0 * (self.u**i) * (self.d ** (t - i))
        # Terminal payoff.
        if self.is_call:
            option[n, : n + 1] = np.maximum(stock[n, : n + 1] - self.K, 0)
        else:
            option[n, : n + 1] = np.maximum(self.K - stock[n, : n + 1], 0)
        # Backward induction.
        for t in range(n - 1, -1, -1):
            for i in range(t + 1):
                option[t, i] = math.exp(-self.r * self.dt) * (
                    self.p * option[t + 1, i + 1] + (1 - self.p) * option[t + 1, i]
                )
        return option[0, 0]


if __name__ == "__main__":
    # Example usage:
    # Price a European call option with S0=100, K=100, r=0.05, sigma=0.2, T=1.0,
    # using 101 steps.
    for k, t in [(90, 1.0), (100, 0.75), (110, 0.5)]:
        lr = LeisenReimerLattice(
            S0=100, K=k, r=0.05, sigma=0.2, q=0.02, T=t, steps=201, is_call=True
        )
        price_call = lr.price_option()
        print(f"S, t = {k, t} European Call Price =", price_call)

        # Price a European put option.
        lp = LeisenReimerLattice(
            S0=100, K=k, r=0.05, sigma=0.2, q=0.02, T=t, steps=201, is_call=False
        )
        price_put = lp.price_option()
        print(f"S, t = {k, t} European Call Price =", price_put)
