
import math
import numpy as np
from typing import Optional, Tuple, Union
from src.stochastic.base_classes.base_stoch_proc import StochasticProcess

class CGMYProcess(StochasticProcess):
    """
    Implements a CGMY (Carr-Geman-Madan-Yor) process with parameters
    C, G, M, and Y, each controlling the Levy measure shape.

    The CGMY process generalizes various Lévy processes.
    """

    def __init__(
        self,
        C: float,
        G: float,
        M: float,
        Y: float,
        start: float = 0.0,
        end: float = 1.0
    ) -> None:
        """
        Parameters
        ----------
        C : float
            Scale parameter (> 0).
        G : float
            Decay rate for upward jumps (> 0).
        M : float
            Decay rate for downward jumps (> 0).
        Y : float
            Tail heaviness, must be < 2 for typical definitions.
        start : float, optional
            Start of the index range.
        end : float, optional
            End of the index range.
        """
        super().__init__(start, end)
        if C <= 0 or G <= 0 or M <= 0:
            raise ValueError("C, G, and M must be positive.")
        if Y >= 2:
            raise ValueError("Y must be less than 2.")

        self.C = float(C)
        self.G = float(G)
        self.M = float(M)
        self.Y = float(Y)

    def __repr__(self) -> str:
        return (f"CGMYProcess(C={self.C}, G={self.G}, M={self.M}, Y={self.Y}, "
                f"index={self.index})")

    def sample(
        self,
        sims: int,
        idx: float,
        shape: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> np.ndarray:
        """
        Sample X_idx for the CGMY Process at a single time idx.

        We do a naive Poisson-approx approach, generating N ~ Poisson(C*idx)
        jumps, each split into upward vs. downward exponentials. For HPC usage,
        consider more advanced methods or chunk-based concurrency.

        Parameters
        ----------
        sims : int
            Number of simulations to generate.
        idx : float
            Time index at which to sample the process.
        shape : int or tuple of int, optional
            Desired output shape of the returned array. If None, returns
            a 1D array of length sims.

        Returns
        -------
        np.ndarray
            Simulated values of the CGMY process at time idx.

        Notes
        -----
        1. This is a very rough approach for demonstration and might be slow or
           inaccurate for large sims or large idx due to naive summation.
        2. For HPC, chunk calls to sample(...) if needed or consider
           advanced Levy area methods or FFT-based approximations.
        """
        if idx < 0:
            raise ValueError("Time idx cannot be negative.")
        if shape is None:
            shape = (sims,)

        rng = np.random.default_rng()
        # Poisson realize N
        N = rng.poisson(lam=self.C * idx, size=sims)

        # scale factor for time exponent in the sense CGMY might do t^(1/Y).
        scale_factor = idx ** (1.0 / self.Y)

        # We must handle upward and downward jumps. We'll do a simple loop approach.
        # For HPC or large sims, chunking or vectorization might be needed.

        out = np.zeros(sims, dtype=float)
        # We gather the maxN to do a 2D approach. Then mask out unneeded parts.
        maxN = N.max()
        if maxN > 0:
            # Up jumps (Exponential(1/G)) and down jumps (Exponential(1/M))
            up_jumps = rng.exponential(scale=1.0/self.G, size=(sims, maxN))
            down_jumps = rng.exponential(scale=1.0/self.M, size=(sims, maxN))

            # We'll mask using broadcasting
            idxs = np.arange(maxN)
            # (N[:,None] > idxs) => True if idxs < N[i]
            # sum up_jumps where that is True
            mask_up = (idxs < N[:, None])
            sum_up = np.sum(up_jumps * mask_up, axis=1)
            sum_down= np.sum(down_jumps * mask_up, axis=1)

            out= sum_up - sum_down

        # multiply by scale factor
        out*= scale_factor

        return out.reshape(shape)

    def graph(self, num_paths: int=1, steps: int=100) -> None:
        """
        Graph sample paths in [start, end] by repeatedly calling sample(...).
        This yields i.i.d. draws at each time, ignoring correlation.
        For HPC correlation or advanced path logic, override simulate_paths().
        """
        super().graph(num_paths=num_paths, steps=steps)
