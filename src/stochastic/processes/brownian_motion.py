#!/usr/bin/env python3
# brownian_motion.py

"""
BrownianMotion Process
======================
A simple Brownian motion (a.k.a. Wiener process) with magnitude (volatility)
factor 'mag', no drift.

Usage
-----
>>> from brownian_motion import BrownianMotion
>>> bm = BrownianMotion(mag=1.0, start=0.0, end=1.0)
>>> samples = bm.sample(sims=10000, idx=0.5)
>>> # samples ~ Normal(0, mag^2 * 0.5)
>>> print(samples.mean(), samples.std())

For entire path stepping from 0..1, you can either:
1) Use the base "simulate_paths" method if your base class calls
   bm.sample(...) at each time step. (This might produce i.i.d. Normal(0, sqrt(t))
   draws at each time, ignoring correlation across time steps.)
2) If you want correlated increments for a full path, override simulate_paths
   with an increment-based approach:
       X_{k+1} = X_k + mag * sqrt(dt)*N(0,1).
"""

import numpy as np
from typing import Optional, Union, Tuple

from src.stochastic.base_classes.base_stoch_proc import StochasticProcess


class BrownianMotion(StochasticProcess):
    """
    A standard Brownian motion W_t with no drift, scaled by 'mag'.

    For a single time idx, W_idx ~ Normal(0, mag^2 * idx).
    """

    def __init__(self, mag: float = 1.0, start: float = 0.0, end: float = 1.0) -> None:
        """
        Initialize a Brownian motion process.

        Parameters
        ----------
        mag : float, optional
            The magnitude (volatility) factor for the Brownian motion.
            If mag=1, this is a standard Wiener process with Var(t)=t.
        start : float, optional
            Start of the index/time domain (default 0.0).
        end : float, optional
            End of the index/time domain (default 1.0).
        """
        super().__init__(start=start, end=end)
        self.mag = float(mag)

    def sample(
        self, sims: int, idx: float, shape: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> np.ndarray:
        """
        Sample Brownian motion at a single time 'idx'.

        W_idx ~ Normal(0, mag^2 * idx).

        Parameters
        ----------
        sims : int
            Number of samples to draw.
        idx : float
            The time or index, presumably in [start, end].
        shape : int or tuple of int, optional
            The desired output shape. If None, returns a 1D array of length sims.

        Returns
        -------
        np.ndarray
            The Brownian motion samples at time idx, drawn from Normal(0, mag^2 * idx).
        """
        if idx < 0:
            raise ValueError("Brownian motion time idx cannot be negative.")
        if shape is None:
            shape = (sims,)

        # variance = mag^2 * idx
        std_dev = np.sqrt(idx) * self.mag
        rng = np.random.default_rng()
        samples = rng.normal(loc=0.0, scale=std_dev, size=shape)
        return samples

    def graph(self, num_paths: int = 1, steps: int = 100) -> None:
        """
        Graph sample paths of the Brownian motion in [self.index[0], self.index[1]].

        This calls the base class's default path-generation logic,
        which might treat each time in the grid as i.i.d. Normal(0, mag^2 * t).
        Note that this yields uncorrelated snapshots across times,
        so it's not a true continuous Brownian path.

        If you want correlated increments, override the entire simulate_paths
        with an increment-based approach.

        Parameters
        ----------
        num_paths : int, optional
            How many paths to plot (default=1).
        steps : int, optional
            Number of steps in [start, end] (default=100).
        """
        super().graph(num_paths=num_paths, steps=steps)
