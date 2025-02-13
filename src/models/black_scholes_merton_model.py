import math
from src.models.base_model import BaseModel
from typing import Any, Dict, Optional


class BlackScholesMertonModel(BaseModel):
    """
    Black-Scholes-Merton model with constant volatility,
    a constant risk-free rate, and an optional continuous dividend yield.
    """

    def __init__(
        self,
        sigma: float,
        risk_free_rate: float,
        dividend_yield: float = 0.0,
        model_name: str = "BlackScholesMerton",
    ) -> None:
        """
        Parameters
        ----------
        sigma : float
            Annualized volatility (e.g., 0.20 = 20%).
        risk_free_rate : float
            Constant risk-free rate (annualized).
        dividend_yield : float, optional
            Continuous dividend yield (annualized), by default 0.0
        model_name : str, optional
            Descriptive name for the model, by default "BlackScholesMerton".
        """
        super().__init__(
            model_name=model_name,
            sigma=float(sigma),
            r=float(risk_free_rate),
            dividend_yield=float(dividend_yield),
        )

    def validate_params(self) -> None:
        """
        Validate that parameters are non-negative where appropriate,
        or raise ValueError otherwise.
        """
        p = self._params
        if p["sigma"] < 0:
            raise ValueError("Volatility (sigma) cannot be negative.")
        if p["r"] < 0:
            # It's possible to have negative rates, but let's impose a simple check:
            raise ValueError(
                "Risk-free rate (r) cannot be negative in this simple example."
            )
        if p["dividend_yield"] < 0:
            raise ValueError("Dividend yield cannot be negative.")

    def price(self, *args, **kwargs) -> float:
        """
        Raises
        ------
        NotImplementedError
            Because, by design, we typically rely on a Technique class
            to handle actual pricing logic in this architecture.
        """
        raise NotImplementedError(
            "BlackScholesMertonModel does not directly price instruments. "
            "Use a Technique (e.g., closed-form, PDE, MC) that references this model."
        )

    def get_stochastic_process(self, *args, **kwargs) -> Any:
        """
        dS = S(r - q) dt + S sigma dW
        This is typically used by Monte Carlo or PDE-based methods.

        Returns
        -------
        SDE class
        """
        p = self._params
        sde_info = {
            "drift": lambda s, t: (p["r"] - p["dividend_yield"]) * s,
            "diffusion": lambda s, t: p["sigma"] * s,
            "risk_free_rate": p["r"],
            "dividend_yield": p["dividend_yield"],
            "sigma": p["sigma"],
        }
        return sde_info

    def char_func(self, S, T):
        """Return the characteristic function for the BSM model.

        Parameters
        ----------
        S   : float : Current price of stock.
        r   : float : Annualized risk-free interest rate, continuously compounded.
        T   : float : Time, in years, until maturity.
        v   : float : Volatility of the stock.
        q   : float : Continous dividend rate.

        Returns
        -------
        res : function : Characteristic function.
        """
        import numpy as np

        p = self._params
        r = p["r"]
        q = p["dividend_yield"]
        v = p["sigma"]
        lnS = np.log(S)
        lnPhi = (
            lambda u: 1j * u * lnS
            + 1j * u * (r - q) * T
            - T * v**2 * (u**2 + 1j * u) / 2
        )
        phi = lambda u: np.exp(lnPhi(u))

        return phi

    def characteristic_function(self, t: float, spot: float = 1.0, **kwargs) -> complex:
        """
        Classic characteristic function for log(S_t) under BSM assumptions
        (risk-neutral measure, with continuous dividend yield q):
          phi(u) = exp(i*u*(ln(S0) + (r-q - 0.5*sigma^2)*t)) * exp(-0.5*sigma^2*u^2 * t)

        Parameters
        ----------
        t : float
            Time in years.
        spot : float, optional
            Current spot price, by default 1.0
        **kwargs :
            Additional parameters if needed.

        Returns
        -------
        complex
            Characteristic function value at time t.
        """

        p = self._params
        r = p["r"]
        q = p["dividend_yield"]
        sigma = p["sigma"]
        u = kwargs.get("u", 0.0)

        # Log of initial spot
        ln_s0 = math.log(spot)
        drift_term = (r - q - 0.5 * sigma**2) * t

        # Exponential part for the log-price
        expo_real = 1j * u * (ln_s0 + drift_term)

        # The variance part: -0.5 * sigma^2 * u^2 * t
        # which contributes to the real part of the exponent
        expo_var = -0.5 * sigma**2 * (u**2) * t

        return complex(math.e ** (expo_real + expo_var))

    def pde_params(self, S: float, t: float = 0.0, **kwargs) -> Dict[str, float]:
        """
        Returns PDE coefficients for a standard Black-Scholes PDE
        at the given S, t. For constant sigma, r, and q:

          diffusion = sigma^2
          drift     = r - q
          rate      = r

        Parameters
        ----------
        S : float
            Current underlying price (unused if sigma is constant,
            but included for general PDE structure).
        t : float, default=0.0
            Current time in years (unused for constant BSM).
        **kwargs :
            Additional PDE parameters if needed.

        Returns
        -------
        Dict[str, float]
            {
              "diffusion": self._params["sigma"]**2,
              "drift": self._params["r"] - self._params["dividend_yield"],
              "rate": self._params["r"]
            }
        """
        p = self._params
        sigma = p["sigma"]
        r = p["r"]
        q = p["dividend_yield"]

        return {"diffusion": sigma * sigma, "drift": r - q, "rate": r}
