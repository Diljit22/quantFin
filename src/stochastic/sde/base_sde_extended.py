import math
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Callable, Tuple


###############################################################################
# Extended Base Class
###############################################################################
class BaseModelExtended(ABC):
    """
    Base class for SDE models with two required methods:
      - sample_paths(T, n_sims, n_steps) -> array (n_sims, n_steps+1)
      - sample_paths_and_derivative(T, n_sims, n_steps) -> (S, dSdS0)

    dSdS0[i,j] = partial(S_{i,j}, S0).
    """

    def __init__(
        self, r: float, q: float, S0: float, random_state: Optional[int] = None
    ):
        if S0 <= 0:
            raise ValueError("Initial spot S0 must be > 0.")
        self.r = r
        self.q = q
        self.S0 = S0
        self._rng = np.random.default_rng(random_state)

    @abstractmethod
    def sample_paths(self, T: float, n_sims: int, n_steps: int) -> np.ndarray:
        """
        Return shape (n_sims, n_steps+1).
        """
        pass

    @abstractmethod
    def sample_paths_and_derivative(
        self, T: float, n_sims: int, n_steps: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (S, dSdS0). Each is shape (n_sims, n_steps+1).
        S[i,j] = path i at time j
        dSdS0[i,j] = partial(S_{i,j}, S0).
        """
        pass

    def __repr__(self) -> str:
        cname = self.__class__.__name__
        return f"{cname}(r={self.r}, q={self.q}, S0={self.S0})"
