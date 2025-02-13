import numpy as np
from src.stochastic.StochasticProcess import StochasticProcess

class HyperbolicLevy(StochasticProcess):
    """
    Implements a Hyperbolic Lévy process.
    The process is characterized by heavy tails and skewness.
    """

    def __init__(self, scale=1, skew=0, start=0, end=1):
        """
        Initialize the Hyperbolic Lévy process.

        Parameters
        ----------
        scale : float : Scale parameter controlling volatility.
        skew  : float : Skewness parameter for asymmetric increments.
        start : float : Start of the index range.
        end   : float : End of the index range.
        """
        super().__init__(start, end)
        self.scale = scale
        self.skew = skew

    def __repr__(self):
        """
        Return a string representation of the process.
        """
        return f"HyperbolicLevy(scale={self.scale}, skew={self.skew}, index={self.index})"

    def sample(self, sims, idx, shape=None):
        """
        Sample X_t for the Hyperbolic Lévy process.

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

        scale_t = self.scale * np.sqrt(idx)  # Scaled variance
        skew_noise = self.skew * np.random.normal(size=shape)
        symmetric_noise = np.random.normal(loc=0, scale=scale_t, size=shape)
        return skew_noise + symmetric_noise
