import math
from typing import Dict, Any


class InterestRateCurve:
    """
    A simple container for an interest rate curve.
    """

    def __init__(self, rates: Dict[float, float]) -> None:
        """
        Initialize the interest rate curve.

        Parameters
        ----------
        rates : dict of float -> float
            Mapping of time (in years) to annualized interest rate.
            e.g. {0.5: 0.02, 1.0: 0.025} means:
            - a 2% 6-month rate
            - a 2.5% 1-year rate
        """
        self._rates = dict(rates)

    def get_rate(self, t: float) -> float:
        """
        Retrieve the interest rate at time t (years).
        Uses a naive nearest-key lookup. Production code would interpolate.

        Parameters
        ----------
        t : float
            Time in years.

        Returns
        -------
        float
            The annualized interest rate corresponding to the nearest provided time.

        Raises
        ------
        ValueError
            If no rates are defined.
        """
        if not self._rates:
            raise ValueError("No rates are defined in the curve.")
        closest_t = min(self._rates.keys(), key=lambda x: abs(x - t))
        return self._rates[closest_t]

    def get_discount_factor(self, t: float) -> float:
        """
        Calculate the discount factor via continuous compounding.

        Parameters
        ----------
        t : float
            Time in years.

        Returns
        -------
        float
            Discount factor for time t.

        Raises
        ------
        ValueError
            If t is negative.
        """
        if t < 0:
            raise ValueError("Time t cannot be negative.")
        if t == 0:
            return 1.0
        r = self.get_rate(t)
        return math.exp(-r * t)
