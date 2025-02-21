"""
european_option.py
==================

Defines the EuropeanOption class, which inherits from BaseOption.
European options can only be exercised at maturity.
"""

from src.instruments.base_option import BaseOption
from dataclasses import dataclass


@dataclass(frozen=True)
class EuropeanOption(BaseOption):
    """
    Represents a European vanilla option that can only be exercised at maturity.

    Parameters
    ----------
    strike : float
        Strike price of the option.
    maturity : float
        Time to maturity in years.
    is_call : bool
        True if the option is a call, False if it is a put.
    """

    def payoff(self, spot_price: float) -> float:
        """
        Compute the payoff at maturity for a European vanilla option.

        Parameters
        ----------
        spot_price : float
            Underlying's spot price at maturity.

        Returns
        -------
        float
            For call options, returns max(spot_price - strike, 0).
            For put options, returns max(strike - spot_price, 0).
        """
        return self.intrinsic_payoff(spot_price)

    def __hashable_state__(self) -> tuple:
        return (self.strike, self.maturity, self.is_call)
