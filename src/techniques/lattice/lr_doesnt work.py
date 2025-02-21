import math
import numpy as np
from scipy.stats import norm
from src.techniques.lattice.base_lattice import BaseLattice


def _J(z, n, gamma=0.0):
    """
    A modified Leisen–Reimer transformation.

    Parameters:
      z : float
          Input (typically d2 or d1).
      n : int
          Number of steps.
      gamma : float, optional
          A calibration parameter (default 0.0). Adjusting gamma changes the effective
          denominator so that the resulting probability p can be nudged up or down.

    Returns:
      float: A value between 0 and 1.
    """
    denom = n + 1.0 / 3.0 + gamma / (n + 1)
    return 0.5 + 0.5 * np.sign(z) * np.sqrt(1 - np.exp(-z * z / denom))


def _compute_ud_p_LR(steps, r, q, sigma, T, S0, K):
    """
    Compute the LR parameters.

    We use:
      u = exp(sigma * sqrt(dt))
      d = 1/u
    and for calls:
      if S0 >= K (in–the–money), set p = 1 - J(d2, steps, gamma)
      else, set p = J(d1, steps, gamma)
    (For puts one would use the complementary transformation.)

    Here gamma is a calibration parameter.
    """
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u

    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Try a small positive gamma (e.g. 0.2) to shift probabilities upward for calls.
    gamma = 0.2
    if S0 >= K:
        p = 1.0 - _J(d2, steps, gamma)
    else:
        p = _J(d1, steps, gamma)
    p = np.clip(p, 1e-8, 1 - 1e-8)
    return float(u), float(d), float(p)


class LeisenReimerLattice(BaseLattice):
    """
    An advanced binomial approach using the Leisen–Reimer method for option pricing,
    calibrated to the Black–Scholes–Merton model.

    Parameters
    ----------
    S0 : float
        Initial stock price.
    K : float
        Strike price.
    r : float
        Risk–free rate.
    sigma : float
        Volatility.
    q : float
        Dividend yield.
    T : float
        Time to maturity.
    steps : int
        Number of time steps.
    is_call : bool
        True for call options; False for put options.
    is_american : bool
        True for American style (early exercise allowed), False for European.
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
        self.discount = math.exp(-self.r * self.dt)

        # Build LR parameters calibrated to BSM.
        self.u, self.d, self.p = _compute_ud_p_LR(
            self.steps, self.r, self.q, self.sigma, self.T, self.S0, self.K
        )
        # Arrays for the lattice.
        self.stock_prices = None
        self.option_values = None

    def build_lattice(self):
        """
        Build 2D arrays (shape (steps+1, steps+1)) for stock prices and option values.
        For each time step t, only the first t+1 elements are used.
        """
        if self.stock_prices is not None and self.option_values is not None:
            return self.stock_prices, self.option_values

        stock = np.zeros((self.steps + 1, self.steps + 1))
        option = np.zeros_like(stock)

        stock[0, 0] = self.S0
        for t in range(1, self.steps + 1):
            stock[t, 0] = stock[t - 1, 0] * self.d
            for i in range(1, t + 1):
                stock[t, i] = stock[t - 1, i - 1] * self.u

        if self.is_call:
            option[self.steps, :] = np.maximum(stock[self.steps, :] - self.K, 0.0)
        else:
            option[self.steps, :] = np.maximum(self.K - stock[self.steps, :], 0.0)

        # Backward induction.
        for t in reversed(range(self.steps)):
            c_up = option[t + 1, 1 : t + 2]  # length = t+1
            c_dn = option[t + 1, 0 : t + 1]  # length = t+1
            continuation = self.discount * (self.p * c_up + (1 - self.p) * c_dn)
            option[t, 0 : t + 1] = continuation

            if self.is_american:
                if self.is_call:
                    exercise = stock[t, 0 : t + 1] - self.K
                else:
                    exercise = self.K - stock[t, 0 : t + 1]
                exercise = np.maximum(exercise, 0.0)
                option[t, 0 : t + 1] = np.maximum(option[t, 0 : t + 1], exercise)

        self.stock_prices = stock
        self.option_values = option
        return stock, option

    def price_option(self) -> float:
        self.build_lattice()
        return self.option_values[0, 0]

    def calc_greeks(self) -> dict:
        """
        Approximate Delta, Gamma, and Theta via finite differences.
        """
        self.build_lattice()
        # Delta from the first step.
        Vu = self.option_values[1, 1]
        Vd = self.option_values[1, 0]
        Su = self.stock_prices[1, 1]
        Sd = self.stock_prices[1, 0]
        Delta = (Vu - Vd) / (Su - Sd)

        # Gamma from two steps.
        Vuu = self.option_values[2, 2]
        Vud = self.option_values[2, 1]
        Vdd = self.option_values[2, 0]
        Suu = self.stock_prices[2, 2]
        Sud = self.stock_prices[2, 1]
        Sdd = self.stock_prices[2, 0]
        partial1 = (Vuu - Vud) / (Suu - Sud)
        partial2 = (Vud - Vdd) / (Sud - Sdd)
        Gamma = (partial1 - partial2) / (0.5 * (Suu - Sdd))

        # Theta approximation.
        V0 = self.option_values[0, 0]
        Vmid = self.option_values[2, 1]
        Theta = (Vmid - V0) / (2.0 * self.dt)

        return dict(Delta=Delta, Gamma=Gamma, Theta=Theta)
