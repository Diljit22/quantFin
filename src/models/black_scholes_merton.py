"""
black_scholes_merton.py
=======================

Defines the BlackScholesMerton class, a subclass of BaseModel.
This model captures the standard Black-Scholes (Merton) assumptions for
equity option pricing, focusing on volatility as its key parameter.

Notes
-----
- Only store the volatility (sigma).

- The characteristic function returned is a function phi(u) that, for a
  given complex argument u, computes the exponential of (i u * log-price).
"""

import math
import cmath
from typing import Callable
import numpy as np

from src.models.base_model import BaseModel


class BlackScholesMerton(BaseModel):
    """
    Black-Scholes-Merton model for option pricing.

    Attributes
    ----------
    _params["sigma"] : float
        The annualized volatility as a decimal (e.g., 0.2 for 20%).
        Must be >= 0.
    """

    def __init__(self, sigma) -> None:
        """
        Initialize a BSM model with a given volatility.

        Parameters
        ----------
        sigma : float
            The annualized volatility.
        """
        super().__init__(model_name="BlackScholesMerton", sigma=sigma)

    @property
    def sigma(self) -> float:
        """
        Returns the stored volatility parameter.
        """
        return self._params["sigma"]

    def validate_params(self) -> None:
        """
        Validate the model parameters.

        Raises
        ------
        ValueError
            If volatility (sigma) is negative.
        """
        sigma = self._params["sigma"]
        if sigma < 0.0:
            raise ValueError("Volatility (sigma) cannot be negative.")

    def characteristic_function(
        self, t: float, spot: float, r: float, q: float
    ) -> Callable[[complex], complex]:
        """
        Return a function phi(u) representing the characteristic function of
        the log of the asset price under the Black-Scholes-Merton model.

        Parameters
        ----------
        t : float
            Time to maturity (in years).
        spot : float
            Current spot price of the underlying asset.
        r : float
            Risk-free interest rate (annualized).
        q : float
            Continuous dividend yield (annualized).

        Returns
        -------
        Callable[[complex], complex]
            A function phi(u) which, when evaluated at a complex number u,
            yields the characteristic function value exp(i u * X_t),
            where X_t is the log of the asset price under BSM dynamics.

        Notes
        -----
        The log of S_t follows a normal distribution with mean:
            ln(spot) + (r - q - sigma^2/2) * t
        and variance:
            sigma^2 * t

        Hence the characteristic function of ln(S_t) is:
            phi(u) = E[e^(i u ln(S_t))]
                    = exp( i u [ln(spot) + (r - q - 0.5 sigma^2) t] - 0.5 sigma^2 t u^2 )
        """
        sigma = self._params["sigma"]
        half_var = 0.5 * sigma * sigma
        drift = math.log(spot) + (r - q - half_var) * t

        def phi(u: complex) -> complex:
            return np.exp((1j * u * drift) - (half_var * t * (u**2)))

        return phi

    def SDE(self, *args, **kwargs) -> None:
        """
        Placeholder for a stochastic differential equation representation:
        dS_t = S_t (r - q) dt + sigma * S_t dW_t

        Raises
        ------
        NotImplementedError
            This method is not implemented in the current design.
        """
        raise NotImplementedError(
            "SDE method is not implemented for BlackScholesMerton."
        )

    def S_t(self, *args, **kwargs) -> None:
        """
        Placeholder for a path simulation or closed-form solution for S(t).

        Raises
        ------
        NotImplementedError
            This method is not implemented in the current design.
        """
        raise NotImplementedError(
            "S_t method is not implemented for BlackScholesMerton."
        )

    def pde(self, S: float, r: float, q: float, K: float, T: float):
        """
        Return the PDE coefficients (A, B, C) for the spatial operator:
            L V = A(S) V'' + B(S) V' - C(S) V.
        For Black–Scholes, these are:
            A = 0.5 * sigma^2 * S^2,
            B = (r - q) * S,
            C = r.

        Parameters
        ----------
        S : float
            Asset price.
        r : float
            Risk-free rate.
        q : float
            Dividend yield.
        sigma : float
            Volatility.
        K : float
            Strike price.
        T : float
            Time to maturity.

        Returns
        -------
        Tuple[float, float, float]
            Coefficients (A, B, C).
        """
        A = 0.5 * self.sigma**2 * S**2
        B = (r - q) * S
        C = r
        return (A, B, C)

    def __repr__(self) -> str:
        """
        String representation for debugging, includes model name and parameters.
        """
        return super().__repr__()

    def __hashable_state__(self) -> tuple:
        return (self.sigma,)
