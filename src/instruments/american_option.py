"""
american_option.py
==================

Defines the AmericanOption class, which inherits from BaseOption.
American options can be exercised any time up to maturity.

Notes
-----
This class is identical to EuropeanOption - difference is for end user.
"""

from src.instruments.base_option import BaseOption


class AmericanOption(BaseOption):
    """
    Represents an American vanilla option. It can be exercised at any time
    up to and including maturity.

    Parameters
    ----------
    strike : float
        The strike price of the option.
    maturity : float
        The time to maturity (in years).
    is_call : bool
        True if the option is a call, False if it is a put.
    """

    def payoff(self, spot_price: float) -> float:
        """
        Computes the payoff if exercised immediately (spot_price known).
        For American options, the actual exercise time may be earlier than
        maturity. This payoff is the standard vanilla payoff function for a
        single point in time.

        Parameters
        ----------
        spot_price : float
            The underlying's spot price at the exercise time.

        Returns
        -------
        float
            max(S - K, 0) for calls, or max(K - S, 0) for puts.
        """
        return self.intrinsic_payoff(spot_price)
