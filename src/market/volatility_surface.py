from typing import Dict, Tuple, Any


class VolatilitySurface:
    """
    Container for a volatility surface.
    """

    def __init__(self, vols: Dict[Tuple[float, float], float]) -> None:
        """
        Initialize the volatility surface.

        Parameters
        ----------
        vols : dict of (float, float) -> float
            Mapping of ((maturity, strike)) -> implied volatility.
            For example, {(1.0, 100): 0.20, (1.0, 110): 0.23}.
        """
        self._vols = dict(vols)

    def get_vol(self, maturity: float, strike: float) -> float:
        """
        Retrieve implied volatility for the given maturity and strike via a
        naive nearest-neighbor approach. Production code would do 2D interpolation.

        Parameters
        ----------
        maturity : float
            Time to maturity in years.
        strike : float
            Strike price.

        Returns
        -------
        float
            Implied volatility (e.g. 0.20 for 20%).

        Raises
        ------
        ValueError
            If the surface has no data.
        """
        if not self._vols:
            raise ValueError("No volatility data in the surface.")
        # Nearest key in the 2D plane
        closest_key = min(
            self._vols.keys(), key=lambda x: abs(x[0] - maturity) + abs(x[1] - strike)
        )
        return self._vols[closest_key]
