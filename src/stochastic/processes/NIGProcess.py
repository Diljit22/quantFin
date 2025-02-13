import numpy as np
from scipy.stats import invgauss
from src.stochastic.StochasticProcess import StochasticProcess

class NIGProcess(StochasticProcess):
    """
    Implements the Normal Inverse Gaussian (NIG) process.

    Combines a Brownian motion subordinated to an Inverse Gaussian process.
    """

    def __init__(self, alpha: float, beta: float, delta: float,
                 mu: float = 0, start: float = 0, end: float = 1):
        """
        Initialize the NIG Process.

        Parameters
        ----------
        alpha : float : Shape parameter, must satisfy alpha > |beta|.
        beta  : float : Skewness parameter.
        delta : float : Scale parameter of the process.
        mu    : float : Location parameter.
        start : float : Start of the index range.
        end   : float : End of the index range.
        """
        if alpha <= abs(beta):
            raise ValueError("Invalid parameters: alpha must be greater than |beta|.")
        super().__init__(start=start, end=end)
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.mu = mu

    def sample(self, sims: int, idx: float, shape: tuple = None) -> np.ndarray:
        """
        Sample the NIG Process.

        Parameters
        ----------
        sims  : int   : Number of simulations to generate.
        idx   : float : Time index at which to sample the process.
        shape : tuple : Desired shape of the output array.

        Returns
        -------
        np.ndarray : Simulated values of the NIG process.
        """
        if shape is None:
            shape = (sims,)

        gamma = np.sqrt(self.alpha**2 - self.beta**2)
        ig_samples = invgauss.rvs(mu=gamma / self.delta,
                                  scale=(self.delta**2) / gamma,
                                  size=shape)

        bm = np.random.normal(loc=self.beta * ig_samples,
                              scale=np.sqrt(ig_samples),
                              size=shape)

        return self.mu * idx + bm
