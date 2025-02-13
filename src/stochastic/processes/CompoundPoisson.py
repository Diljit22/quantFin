import numpy as np
from src.stochastic.base_classes.base_stoch_proc import StochasticProcess

class CompoundPoisson(StochasticProcess):
    """
    Compound Poisson process with lognormal jumps. 
    X_t = sum of lognormal i.i.d. jumps triggered by a Poisson(lam * t),
    scaled by 'mag'.
    """

    def __init__(
        self,
        lam: float,
        lognorm_mean: float = 0.0,
        lognorm_dev: float = 1.0,
        mag: float = 1.0,
        start: float = 0.0,
        end: float = 1.0
    ) -> None:
        """
        Parameters
        ----------
        lam : float
            Intensity (rate) of the Poisson process (>0).
        lognorm_mean : float, optional
            Mean of the lognormal jump distribution (default=0).
        lognorm_dev : float, optional
            Std dev of the lognormal jump distribution (default=1).
        mag : float, optional
            Magnitude scaling factor (default=1).
        start : float, optional
            Start time (default=0).
        end : float, optional
            End time (default=1).
        """
        super().__init__(start, end)
        if lam < 0:
            raise ValueError("Lam must be nonnegative.")
        self.lam = float(lam)
        self.lognorm_mean = float(lognorm_mean)
        self.lognorm_dev  = float(lognorm_dev)
        self.mag = float(mag)

    def __repr__(self) -> str:
        params = (f"lam={self.lam}, logNormMean={self.lognorm_mean}, "
                  f"logNormDev={self.lognorm_dev}, mag={self.mag}, index={self.index}")
        return f"CompoundPoisson({params})"

    def sample(
        self,
        sims: int,
        idx: float,
        shape = None
    ) -> np.ndarray:
        """
        Generate samples X_idx of the compound Poisson process at time idx.

        N ~ Poisson(lam * idx).
        Then X_idx = mag * sum_{k=1..N} Y_k, where Y_k ~ Lognormal(lognorm_mean, lognorm_dev).

        Parameters
        ----------
        sims : int
            Number of simulations to draw.
        idx : float
            Time index at which to sample the process (>=0).
        shape : int or tuple of int, optional
            Desired shape for the output array.

        Returns
        -------
        np.ndarray
            Samples from the Compound Poisson process at time idx, shape=shape or (sims,).

        Notes
        -----
        - For HPC, we do a naive loop combining Poisson draws with lognormal sums.
          If `sims` is large, you might consider chunk-based concurrency or a
          more advanced approach to skip the per-sample loop.
        - The sum-of-jumps is computed by partial indexing in the jumpRealizations array.
        """
        if idx < 0:
            raise ValueError("Time idx cannot be negative.")
        if shape is None:
            shape = (sims,)

        rng = np.random.default_rng()
        # Poisson for each sample
        N_arr = rng.poisson(self.lam* idx, size=sims)

        # The total number of jumps across all samples
        total_jumps = N_arr.sum()
        if total_jumps == 0:
            # no jumps => process =0
            return np.zeros(shape, dtype=float)

        # lognormal jumps for entire batch
        jump_realizations = rng.lognormal(
            mean=self.lognorm_mean,
            sigma=self.lognorm_dev,
            size=total_jumps
        )

        out = np.zeros(sims, dtype=float)
        # We do a partial indexing approach
        start_idx= 0
        for i, n_jumps in enumerate(N_arr):
            if n_jumps>0:
                out[i] = jump_realizations[start_idx:start_idx+n_jumps].sum()
            start_idx+= n_jumps

        # scale by 'mag'
        out*= self.mag
        return out.reshape(shape)

    def graph(self, num_paths: int=1, steps: int=100) -> None:
        """
        Plot the process in [start,end] by i.i.d. calls to sample(...).
        This is an uncorrelated approach. For correlated increments or
        sub-interval logic, override 'simulate_paths'.
        """
        super().graph(num_paths=num_paths, steps=steps)
