import numpy as np
from src.stochastic.base_classes.base_stoch_proc import StochasticProcess

class DeterministicProcess(StochasticProcess):
    """
    A deterministic process: X_t = rate * t.
    If you want a different function in time, override sample(...) or
    create a new child class.
    """

    def __init__(
        self,
        rate: float = 1.0,
        start: float = 0.0,
        end: float = float("inf")
    ) -> None:
        """
        Parameters
        ----------
        rate : float
            The deterministic rate (e.g., drift slope).
        start : float
            Start of the index range.
        end : float
            End of the index range (default=∞).
        """
        super().__init__(start, end)
        self.rate = float(rate)

    def __repr__(self) -> str:
        return (f"DeterministicProcess(rate={self.rate}, "
                f"index=[{self.index[0]}, {self.index[1]}])")

    def sample(
        self,
        sims: int,
        idx: float,
        shape=None,
    ) -> np.ndarray:
        """
        Sample the deterministic process, X_idx = rate * idx.

        Parameters
        ----------
        sims : int
            Number of simulations.
        idx : float
            Time index (>=0).
        shape : int or tuple of int, optional
            Output shape (default: (sims,)).

        Returns
        -------
        np.ndarray
            Deterministic values of shape=shape or (sims,).
        """
        if idx < 0:
            raise ValueError("Time idx cannot be negative.")
        if shape is None:
            shape = (sims,)

        out = np.full(shape, self.rate * idx, dtype=float)
        return out

    def graph(self, num_paths: int=1, steps: int=100) -> None:
        """
        Plot the deterministic line over [start,end].
        We can just call the parent which does repeated sample(...) calls.
        """
        super().graph(num_paths=num_paths, steps=steps)