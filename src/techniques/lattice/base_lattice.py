"""
base_lattice.py

An abstract base class for lattice-based option pricing.
We will extend this in leisen_reimer_lattice.py for a more advanced approach.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseLattice(ABC):
    """
    Abstract base for lattice methods. Provides the skeleton:
      - build_lattice() -> (stock_prices, option_values)
      - price_option() -> float
      - calc_greeks() -> dict
      - interpolate_option_value(S, t) -> float
    """

    @abstractmethod
    def build_lattice(self):
        pass

    @abstractmethod
    def price_option(self) -> float:
        pass

    @abstractmethod
    def calc_greeks(self) -> dict:
        pass

    def interpolate_option_value(self, S: float, t: float) -> float:
        """
        Return the option value for arbitrary underlying S, time t within [0, T].
        This can be done by searching for the appropriate step and node in the lattice
        and doing local interpolation if needed.
        """
        pass
