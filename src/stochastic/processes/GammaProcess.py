import numpy as np
from src.stochastic.StochasticProcess import StochasticProcess

class GammaProcess(StochasticProcess):
    """
    Implements a Gamma Process X_t with parameters a (shape) and b (rate).

    X_t represents the sum of increments drawn from a Gamma distribution.
    """

    def __init__(self, shape: float = None, rate: float = None,
                 theta: float = None, lam: float = None,
                 mag = 1,
                 start: float = 0, end: float = 1):
        """
        Initialize the Gamma Process.

        Parameters
        ----------
        shape : float, optional : Shape parameter (a) of the Gamma distribution.
        rate  : float, optional : Rate parameter (b) of the Gamma distribution.
        theta : float, optional : Mean of the Gamma process increments per unit time.
        lam   : float, optional : Scale parameter of the Gamma process.
        start : float : Start of the index range.
        end   : float : End of the index range.

        Notes
        -----
        If `theta` and `lam` are provided, `shape` and `rate` are computed as:
            shape = theta / lam
            rate = 1 / lam
        """
        super().__init__(start, end)
        self.mag= mag
        if theta is not None and lam is not None:
            self.shape = theta / lam
            self.rate = 1 / lam
            self.theta = theta
            self.lam = lam
        elif shape is not None and rate is not None:
            self.shape = shape
            self.rate = rate
            self.theta = 1/self.shape
            self.lam = 1/self.rate
        else:
            raise ValueError("Either (shape, rate) or (theta, lam) must be provided.")
    def __repr__(self):
        """
        Return a string representation of the process.
        """
        return f"GammaProcess(shape={self.shape}, rate={self.rate}, index={self.index})"

    def sample(self, sims: int, idx: float, shape: tuple = None) -> np.ndarray:
        """
        Sample X_t for the Gamma Process.

        Parameters
        ----------
        sims  : int   : Number of simulations to generate.
        idx   : float : Time index at which to sample the process.
        shape : tuple : Desired shape of the output array.

        Returns
        -------
        np.ndarray : Simulated values of the Gamma Process at time idx.
        """
#        if idx < self.index[0] or idx > self.index[1]:
#            raise ValueError(f"Index {idx} is out of bounds: {self.index}.")

        if shape is None:
            shape = (sims,)

        scale = 1 / self.rate  # Scale parameter for the Gamma distribution
        return self.mag * np.random.gamma(shape=self.shape * idx, scale=scale, size=shape)
