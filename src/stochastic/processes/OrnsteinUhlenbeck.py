import numpy as np
from src.stochastic.StochasticProcess import StochasticProcess

class OrnsteinUhlenbeck(StochasticProcess):
    """
    Implements an Ornstein-Uhlenbeck process.
    """

    def __init__(self, theta, mu, sigma, start=0, end=1):
        """
        Initialize the Ornstein-Uhlenbeck process.

        Parameters
        ----------
        theta : float : Mean reversion rate.
        mu    : float : Long-term mean.
        sigma : float : Volatility parameter.
        start : float : Start of the index range.
        end   : float : End of the index range.
        """
        super().__init__(start, end)
        self.theta = theta
        self.mu = mu
        self.sigma = sigma

    def __repr__(self):
        """
        Return a string representation of the process.
        """
        return (f"OrnsteinUhlenbeck(theta={self.theta}, mu={self.mu}, "
                f"sigma={self.sigma}, index={self.index})")

    def sample(self, sims, idx, shape=None):
        """
        Sample X_t for the Ornstein-Uhlenbeck process.

        Parameters
        ----------
        sims  : int   : Number of simulations to generate.
        idx   : float : Time index at which to sample the process.
        shape : tuple : Desired shape of the output array.

        Returns
        -------
        np.ndarray : Simulated values of the process at time idx.
        """
        if shape is None:
            shape = (sims,)

        # Simplified mean and variance for the process
        mean = self.mu * np.exp(-self.theta * idx)
        variance = self.sigma ** 2 * (1 - np.exp(-2 * self.theta * idx)) / (2 * self.theta)

        # Sample from the normal distribution
        return np.random.normal(loc=mean, scale=np.sqrt(variance), size=shape)
