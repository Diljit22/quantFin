"""
sde_base.py
===========

Defines the abstract base class for 1D SDE-based models, enforcing
a sample_paths(...) method. Also includes any common utility or checks.
"""

import abc
import numpy as np
from typing import Optional


class BaseSDEModel(abc.ABC):
    """
    Abstract base class for 1D (possibly extended) SDE models used in
    Monte Carlo option pricing.

    Subclasses must implement:
        sample_paths(T, n_sims, n_steps) -> np.ndarray

    Attributes
    ----------
    r : float
        Risk-free interest rate (annual, continuous compounding).
    q : float
        Continuous dividend yield.
    S0 : float
        Initial spot price at t=0.
    random_state : Optional[int]
        Seed for reproducible random draws.
    """

    def __init__(
        self, r: float, q: float, S0: float, random_state: Optional[int] = None
    ):
        if S0 <= 0.0:
            raise ValueError("Initial spot S0 must be > 0.")
        self.r = r
        self.q = q
        self.S0 = S0
        self._rng = np.random.default_rng(random_state)

    @abc.abstractmethod
    def sample_paths(self, T: float, n_sims: int, n_steps: int) -> np.ndarray:
        """
        Generate Monte Carlo paths for the price process from t=0 to t=T.

        Parameters
        ----------
        T : float
            Total time horizon in years.
        n_sims : int
            Number of simulated paths.
        n_steps : int
            Number of discrete time steps.

        Returns
        -------
        np.ndarray
            Shape (n_sims, n_steps+1). Each row is one simulated path over time,
            from index 0 (t=0) to index n_steps (t=T).
        """
        pass

    def __repr__(self) -> str:
        cname = self.__class__.__name__
        return f"{cname}(r={self.r}, q={self.q}, S0={self.S0})"
