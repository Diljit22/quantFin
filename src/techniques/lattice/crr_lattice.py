"""
crr_lattice.py

Implements a CRR (Cox-Ross-Rubinstein) binomial lattice for option pricing,
both American and European. Computes Delta, Gamma, Theta in a single pass
without multiple re-runs. Code uses NumPy vectorization for speed.

Classes:
--------
- BaseLattice (abstract)
- CRRLattice (concrete)

Usage Example:
--------------
if __name__ == '__main__':
    crr = CRRLattice(S0=100, K=100, r=0.05, sigma=0.2, q=0.0, T=1.0, steps=100, is_call=True, is_american=False)
    price = crr.price_option()
    greeks = crr.calc_greeks()
    print("CRR Price =", price)
    print("Greeks =", greeks)
"""

import numpy as np

from src.techniques.lattice.base_lattice import BaseLattice


class CRRLattice(BaseLattice):
    """
    Cox-Ross-Rubinstein binomial model.

    Parameters
    ----------
    S0 : float
        Initial underlying price.
    K : float
        Strike price.
    r : float
        Risk-free interest rate (annual).
    sigma : float
        Volatility.
    q : float
        Continuous dividend yield.
    T : float
        Time to maturity (years).
    steps : int
        Number of binomial steps.
    is_call : bool
        True for call, False for put.
    is_american : bool
        True if American exercise, else European.

    Notes on Vectorization:
      We'll store a 2D array stock_prices[step, node].
      Then do a backward induction for option_values[step, node].
    """

    def __init__(
        self,
        S0: float,
        K: float,
        r: float,
        sigma: float,
        q: float,
        T: float,
        steps: int,
        is_call: bool,
        is_american: bool,
    ):
        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.q = q
        self.T = T
        self.steps = steps
        self.is_call = is_call
        self.is_american = is_american

        # Precompute
        self.dt = self.T / self.steps
        # CRR up/down
        self.u = np.exp(self.sigma * np.sqrt(self.dt))
        self.d = 1.0 / self.u
        self.discount = np.exp(-self.r * self.dt)
        # risk-neutral prob
        self.p = (np.exp((self.r - self.q) * self.dt) - self.d) / (self.u - self.d)

    def build_lattice(self):
        """
        Build 2D arrays:
          stock_prices: shape (steps+1, steps+1), stock_prices[t, i] = price at time t, node i
          option_values: same shape, filled in backward induction.

        Return (stock_prices, option_values).
        """
        stock_prices = np.zeros((self.steps + 1, self.steps + 1), dtype=float)
        stock_prices[0, 0] = self.S0

        for t in range(1, self.steps + 1):
            # node i from 0..t
            # vector form: stock_prices[t, :t+1] = stock_prices[t-1, :t] * [u or d]
            # but we do a direct approach:
            stock_prices[t, 0] = stock_prices[t - 1, 0] * self.d
            for i in range(1, t + 1):
                stock_prices[t, i] = stock_prices[t - 1, i - 1] * self.u

        # Now option_values for backward
        option_values = np.zeros_like(stock_prices)
        # terminal payoff
        if self.is_call:
            option_values[self.steps, :] = np.maximum(
                stock_prices[self.steps, :] - self.K, 0.0
            )
        else:
            option_values[self.steps, :] = np.maximum(
                self.K - stock_prices[self.steps, :], 0.0
            )

        for t in reversed(range(self.steps)):
            continuation = self.discount * (
                self.p * option_values[t + 1, 1 : t + 2]
                + (1 - self.p) * option_values[t + 1, 0 : t + 1]
            )
            option_values[t, 0 : t + 1] = continuation

            if self.is_american:
                if self.is_call:
                    exercise_val = np.maximum(stock_prices[t, 0 : t + 1] - self.K, 0.0)
                else:
                    exercise_val = np.maximum(self.K - stock_prices[t, 0 : t + 1], 0.0)
                option_values[t, 0 : t + 1] = np.maximum(
                    option_values[t, 0 : t + 1], exercise_val
                )

        return stock_prices, option_values

    def price_option(self) -> float:
        """
        Return the option price at the root node.
        """
        _, option_values = self.build_lattice()
        return option_values[0, 0]

    def calc_greeks(self) -> dict:
        """
        Compute Delta, Gamma, Theta at the root using standard binomial approximations:

        Delta ~ (V[u] - V[d]) / (S[u] - S[d])
        Gamma ~ second difference from same step
        Theta ~ (V[mid_step] - V[0]) / dt ?

        We do a typical approach: look at the next step in the tree:
          node u => (option_values[1,1], stock_prices[1,1])
          node d => (option_values[1,0], stock_prices[1,0])
        Then do standard formula at t=0.
        """
        stock_prices, option_values = self.build_lattice()
        # next step up/down
        Vu = option_values[1, 1]
        Vd = option_values[1, 0]
        Su = stock_prices[1, 1]
        Sd = stock_prices[1, 0]

        # Delta
        Delta = (Vu - Vd) / (Su - Sd)

        # Gamma => we look at t=2 nodes: uu => [2,2], ud => [2,1], dd => [2,0]
        # typical formula: gamma = ((Vuu - Vud)/(Suu - Sud) - (Vud - Vdd)/(Sud - Sdd)) / (0.5*(Suu - Sdd))
        # we'll do a quick approach:
        Vuu = option_values[2, 2]
        Vud = option_values[2, 1]
        Vdd = option_values[2, 0]
        Suu = stock_prices[2, 2]
        Sud = stock_prices[2, 1]
        Sdd = stock_prices[2, 0]

        # partial_1 = (Vuu - Vud)/(Suu - Sud)
        # partial_2 = (Vud - Vdd)/(Sud - Sdd)
        # center = 0.5*(Suu - Sdd)
        partial_1 = (Vuu - Vud) / (Suu - Sud)
        partial_2 = (Vud - Vdd) / (Sud - Sdd)
        Gamma = (partial_1 - partial_2) / (0.5 * (Suu - Sdd))

        # Theta => we do approximate:
        # Theta ~ (option_values[2,1] - option_values[0,0]) / (2 * dt)
        # i.e. 2 steps in time. or we do a single-step approach
        # a common approach: (Vud - V[0])/(2 dt)
        V0 = option_values[0, 0]
        Vmid = option_values[2, 1]  # the "middle" node at t=2 => a typical approach
        Theta = (Vmid - V0) / (2.0 * self.dt)

        return dict(Delta=Delta, Gamma=Gamma, Theta=Theta)
