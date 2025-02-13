import numpy as np
from src.stochastic.StochasticProcess import StochasticProcess

class MeixnerProcess(StochasticProcess):
    """
    Implements the Meixner Process with parameters a, b, and d.

    The Meixner Process is a Lévy process suitable for modeling skewness and kurtosis.
    """

    def __init__(self, a, b, d, start=0, end=1):
        """
        Initialize the Meixner Process.

        Parameters
        ----------
        a    : float : Scale parameter (> 0).
        b    : float : Skewness parameter (-π < b < π).
        d    : float : Kurtosis parameter (> 0).
        start : float : Start of the index range.
        end   : float : End of the index range.
        """
        super().__init__(start, end)
        if a <= 0 or d <= 0:
            raise ValueError("a and d must be positive.")
        if not (-np.pi < b < np.pi):
            raise ValueError("b must be in the range (-π, π).")

        self.a = a
        self.b = b
        self.d = d

    def __repr__(self):
        """
        Return a string representation of the process.
        """
        return f"MeixnerProcess(a={self.a}, b={self.b}, d={self.d}, index={self.index})"

    def sample(self, sims, idx, shape=None):
        """
        Sample X_t for the Meixner Process.

        Parameters
        ----------
        sims  : int   : Number of simulations to generate.
        idx   : float : Time index at which to sample the process.
        shape : tuple : Desired shape of the output array.

        Returns
        -------
        np.ndarray : Simulated values of the Meixner Process at time idx.
        """
        if shape is None:
            shape = (sims,)

        # Time-scaling factor
        scale_factor = self.a * idx

        # Sample from the Meixner distribution
        U = np.random.uniform(-np.pi / 2, np.pi / 2, size=shape)
        V = np.random.exponential(1, size=shape)

        # Compute components for the Meixner distribution
        cos_b = np.cos(self.b)
        sin_b = np.sin(self.b)
        tan_U = np.tan(U)
        numerator = cos_b + sin_b * tan_U
        denominator = np.cos(U)

        # Avoid division by zero or negative arguments for log
        numerator = np.maximum(numerator, 1e-10)  # Prevent very small values
        denominator = np.maximum(denominator, 1e-10)  # Prevent division by zero

        log_argument = numerator / denominator
        log_argument = np.maximum(log_argument, 1e-10)  # Ensure positivity for log

        # Meixner distribution formula
        samples = (2 * V / (np.pi * self.d * sin_b)) * np.log(log_argument)

        return scale_factor * samples
