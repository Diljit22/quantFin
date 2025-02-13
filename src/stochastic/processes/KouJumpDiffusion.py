import numpy as np
from src.stochastic.StochasticProcess import StochasticProcess

class KouJumpDiffusion(StochasticProcess):
    """
    Implements the Kou Jump Diffusion process.

    Combines a Brownian motion with drift and a Compound Poisson process
    with double exponential jumps.
    """

    def __init__(self, drift: float, volatility: float, lam: float,
                 p: float, eta1: float, eta2: float, start: float = 0, end: float = 1):
        """
        Initialize the Kou Jump Diffusion Process.

        Parameters
        ----------
        drift     : float : Drift of the Brownian motion component.
        volatility: float : Volatility of the Brownian motion component.
        lam       : float : Intensity of the Poisson process.
        p         : float : Probability of upward jump.
        eta1      : float : Rate of upward exponential jumps.
        eta2      : float : Rate of downward exponential jumps.
        start     : float : Start of the index range.
        end       : float : End of the index range.
        """
        super().__init__(start=start, end=end)
        self.drift = drift
        self.volatility = volatility
        self.lam = lam
        self.p = p
        self.eta1 = eta1
        self.eta2 = eta2

    def sample(self, sims: int, idx: float, shape: tuple = None) -> np.ndarray:
        """
        Sample the Kou Jump Diffusion Process.

        Parameters
        ----------
        sims  : int   : Number of simulations to generate.
        idx   : float : Time index at which to sample the process.
        shape : tuple : Desired shape of the output array.

        Returns
        -------
        np.ndarray : Simulated values of the Kou Jump Diffusion process.
        """
        if shape is None:
            shape = (sims,)

        # Brownian motion component
        bm = np.random.normal(loc=self.drift * idx,
                              scale=self.volatility * np.sqrt(idx),
                              size=shape)

        # Poisson process component
        poisson_count = np.random.poisson(self.lam * idx, size=shape)
        jumps = np.zeros(shape)
        for i in range(shape[0]):
            num_jumps = poisson_count[i]
            if num_jumps > 0:
                upward_jumps = np.random.exponential(1 / self.eta1, size=num_jumps)
                downward_jumps = np.random.exponential(1 / self.eta2, size=num_jumps)
                direction = np.random.choice([1, -1], size=num_jumps, p=[self.p, 1 - self.p])
                jumps[i] = np.sum(direction * upward_jumps - (1 - direction) * downward_jumps)

        return bm + jumps
