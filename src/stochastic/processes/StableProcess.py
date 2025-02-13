import numpy as np
from src.stochastic.StochasticProcess import StochasticProcess

class StableProcess(StochasticProcess):
    """
    Implements a Stable Process with parameters alpha, beta, gamma, and delta.
    """

    def __init__(self, alpha, beta, gamma, delta, start=0, end=1):
        """
        Initialize the Stable Process.

        Parameters
        ----------
        alpha : float : Stability index (0 < alpha <= 2).
        beta  : float : Skewness parameter (-1 <= beta <= 1).
        gamma : float : Scale parameter (> 0).
        delta : float : Location parameter.
        start : float : Start of the index range.
        end   : float : End of the index range.
        """
        super().__init__(start, end)
        if not (0 < alpha <= 2):
            raise ValueError("Alpha must be in the range (0, 2].")
        if not (-1 <= beta <= 1):
            raise ValueError("Beta must be in the range [-1, 1].")
        if gamma <= 0:
            raise ValueError("Gamma must be positive.")

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def __repr__(self):
        """
        Return a string representation of the process.
        """
        return (f"StableProcess(alpha={self.alpha}, beta={self.beta}, "
                f"gamma={self.gamma}, delta={self.delta}, index={self.index})")

    def sample(self, sims, idx, shape=None):
        """
        Sample X_t for the Stable Process.

        Parameters
        ----------
        sims  : int   : Number of simulations to generate.
        idx   : float : Time index at which to sample the process.
        shape : tuple : Desired shape of the output array.

        Returns
        -------
        np.ndarray : Simulated values of the Stable Process at time idx.
        """
        if shape is None:
            shape = (sims,)

        # Time scaling for Lévy process
        scaled_gamma = self.gamma * idx ** (1 / self.alpha)

        # Chambers-Mallows-Stuck method
        U = np.random.uniform(-np.pi / 2, np.pi / 2, size=shape)
        W = np.random.exponential(scale=1, size=shape)

        factor = np.tan(np.pi * self.alpha / 2) if self.alpha != 1 else -np.log(W)
        X = (np.sin(self.alpha * U) /
             (np.cos(U) ** (1 / self.alpha))) * \
            ((np.cos(U - self.alpha * U) / W) ** ((1 - self.alpha) / self.alpha))

        if self.alpha == 1:
            X += self.beta * np.log(W)

        return scaled_gamma * X + self.delta
