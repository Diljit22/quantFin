"""
topm_lattice.py

Trinomial Option Pricing Model (TOPM) for European-style options.
This implementation uses a recombining trinomial tree.

Model Setup:
-------------
For each time step Δt = T/steps:
  - dx = σ * sqrt(3 * Δt)
  - u = exp(dx), d = exp(-dx), and m = 1.0.
  
Risk-neutral probabilities are defined as:
  pu = 1/6 + ((r - q - 0.5σ²) * sqrt(Δt)) / (2σ√3)
  pm = 2/3
  pd = 1/6 - ((r - q - 0.5σ²) * sqrt(Δt)) / (2σ√3)

These probabilities sum to 1 and ensure non-negative values when parameters are chosen appropriately.

Usage Example:
--------------
if __name__=='__main__':
    topm = TOPMLattice(S0=100, K=100, r=0.05, sigma=0.2, q=0.0, T=1.0, steps=100,
                       is_call=True, is_american=False)
    price = topm.price_option()
    greeks = topm.calc_greeks()
    print("TOPM Price =", price)
    print("Greeks =", greeks)
"""

import numpy as np
from src.techniques.lattice.base_lattice import BaseLattice


class TOPMLattice(BaseLattice):
    """
    Trinomial Option Pricing Model (TOPM) using a recombining trinomial tree.

    For each time step Δt:
      - dx = σ * sqrt(3 * Δt)
      - u = exp(dx), d = exp(-dx), m = 1.0

    Risk-neutral probabilities:
      pu = 1/6 + ((r - q - 0.5σ²) * sqrt(Δt)) / (2σ√3)
      pm = 2/3
      pd = 1/6 - ((r - q - 0.5σ²) * sqrt(Δt)) / (2σ√3)

    The model builds a stock price lattice and computes the option value via backward induction.
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

        self.dt = self.T / self.steps

        # Use dx = σ * sqrt(3 * dt) so that the tree recombines.
        dx = self.sigma * np.sqrt(3 * self.dt)
        self.u = np.exp(dx)
        self.d = np.exp(-dx)
        self.m = 1.0

        # Risk-neutral probabilities for the trinomial tree:
        self.pu = 1 / 6 + (
            (self.r - self.q - 0.5 * self.sigma**2) * np.sqrt(self.dt)
        ) / (2 * self.sigma * np.sqrt(3))
        self.pm = 2 / 3
        self.pd = 1 / 6 - (
            (self.r - self.q - 0.5 * self.sigma**2) * np.sqrt(self.dt)
        ) / (2 * self.sigma * np.sqrt(3))

        self.discount = np.exp(-self.r * self.dt)

    def build_lattice(self):
        """
        Build the stock price and option value lattices.
        The lattice is represented as a 2D array with dimensions (steps+1) x (2*steps+1).
        The central column (index = steps) at time 0 holds the initial stock price.
        """
        n_nodes = 2 * self.steps + 1  # Maximum nodes at maturity
        stock_prices = np.zeros((self.steps + 1, n_nodes))
        option_values = np.zeros_like(stock_prices)

        mid_index = self.steps  # Place the initial stock price in the middle
        stock_prices[0, mid_index] = self.S0

        # Forward induction: populate the stock price tree
        for t in range(1, self.steps + 1):
            for i in range(mid_index - t, mid_index + t + 1):
                if stock_prices[t - 1, i] != 0.0:
                    Scenter = stock_prices[t - 1, i]
                    # For each node, assign the down, middle, and up moves.
                    stock_prices[t, i - 1] = Scenter * self.d
                    stock_prices[t, i] = Scenter * self.m
                    stock_prices[t, i + 1] = Scenter * self.u

        # Terminal payoff: max(S-K,0) for call, max(K-S,0) for put.
        if self.is_call:
            payoff = np.maximum(stock_prices[self.steps, :] - self.K, 0.0)
        else:
            payoff = np.maximum(self.K - stock_prices[self.steps, :], 0.0)
        option_values[self.steps, :] = payoff

        # Backward induction: compute option values at earlier nodes
        for t in reversed(range(self.steps)):
            for i in range(mid_index - t, mid_index + t + 1):
                if stock_prices[t, i] != 0.0:
                    cont_val = self.discount * (
                        self.pu * option_values[t + 1, i + 1]
                        + self.pm * option_values[t + 1, i]
                        + self.pd * option_values[t + 1, i - 1]
                    )
                    if self.is_american:
                        # For American options, check for early exercise.
                        if self.is_call:
                            exercise = max(stock_prices[t, i] - self.K, 0.0)
                        else:
                            exercise = max(self.K - stock_prices[t, i], 0.0)
                        option_values[t, i] = max(cont_val, exercise)
                    else:
                        option_values[t, i] = cont_val

        return stock_prices, option_values

    def price_option(self) -> float:
        _, option_vals = self.build_lattice()
        # The root is at time 0 and the middle column.
        return option_vals[0, self.steps]

    def calc_greeks(self) -> dict:
        """
        Approximate Greeks using finite differences from the lattice:
          - Delta: first-step difference
          - Gamma: second-step difference
          - Theta: difference between a node two steps ahead and the root, divided by time
        """
        stock_prices, option_vals = self.build_lattice()
        root_idx = self.steps  # middle index at t=0
        V0 = option_vals[0, root_idx]

        # Delta approximation using first step (up and down moves)
        Vu = option_vals[1, root_idx + 1]
        Vd = option_vals[1, root_idx - 1]
        Su = stock_prices[1, root_idx + 1]
        Sd = stock_prices[1, root_idx - 1]
        Delta = (Vu - Vd) / (Su - Sd)

        # Gamma approximation using nodes two steps ahead.
        Vuu = option_vals[2, root_idx + 2]
        Vum = option_vals[2, root_idx + 1]
        Vmm = option_vals[2, root_idx]
        Vmd = option_vals[2, root_idx - 1]
        Vdd = option_vals[2, root_idx - 2]
        Suu = stock_prices[2, root_idx + 2]
        Sum = stock_prices[2, root_idx + 1]
        Smm = stock_prices[2, root_idx]
        partial1 = (Vuu - Vum) / (Suu - Sum)
        partial2 = (Vum - Vmm) / (Sum - Smm)
        center = 0.5 * (Suu - Smm)
        Gamma = (partial1 - partial2) / center

        # Theta approximation: difference between the second time step and the root.
        V2mid = option_vals[2, root_idx]
        Theta = (V2mid - V0) / (2.0 * self.dt)

        return dict(Delta=Delta, Gamma=Gamma, Theta=Theta)


if __name__ == "__main__":
    # Simple test example.
    topm = TOPMLattice(
        S0=100.0,
        K=100.0,
        r=0.05,
        sigma=0.2,
        q=0.02,
        T=1.0,
        steps=100,
        is_call=True,
        is_american=False,
    )
    price = topm.price_option()
    greeks = topm.calc_greeks()
    print("TOPM Price =", price)
    print("Greeks =", greeks)
