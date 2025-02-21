"""
base_option.py
==============

Defines the abstract base class for options. Other option types (European,
American, Bermudan) inherit from this class.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class BaseOption(ABC):
    """
    Abstract base class for a vanilla option.

    Parameters
    ----------
    strike : float
        Strike price of the option (must be > 0).
    maturity : float
        Time to maturity in years (must be > 0).
    is_call : bool
        True for a call, False for a put.
    """

    strike: float
    maturity: float
    is_call: bool

    def __post_init__(self) -> None:
        """
        Perform post-initialization validation.

        Raises
        ------
        ValueError
            If `strike` or `maturity` is not positive.
        """
        if self.strike <= 0:
            raise ValueError("Strike price must be positive.")
        if self.maturity <= 0:
            raise ValueError("Time to maturity must be positive.")

    @property
    def option_type(self) -> str:
        """
        Return the option type as a string.

        Returns
        -------
        str
            "Call" if `is_call` is True, otherwise "Put".
        """
        return "Call" if self.is_call else "Put"

    def intrinsic_payoff(self, spot_price: float) -> float:
        """
        Compute the intrinsic payoff for a vanilla call or put.

        The intrinsic payoff is defined as max(spot_price - strike, 0) for calls,
        and max(strike - spot_price, 0) for puts.

        Parameters
        ----------
        spot_price : float
            Underlying's spot price at exercise/maturity.

        Returns
        -------
        float
            The intrinsic payoff.
        """
        intrin_val = spot_price - self.strike
        return max(intrin_val, 0.0) if self.is_call else max(-intrin_val, 0.0)

    @abstractmethod
    def payoff(self, spot_price: float) -> float:
        """
        Compute the payoff at exercise.

        Subclasses should override this method. The default implementation may
        simply call `intrinsic_payoff` for standard options.

        Parameters
        ----------
        spot_price : float
            Underlying's spot price at the time of exercise/maturity.

        Returns
        -------
        float
            The payoff value.
        """
        pass

    def with_strike(self, new_strike: float) -> "BaseOption":
        """
        Create a new option instance with an updated strike price.

        Parameters
        ----------
        new_strike : float
            The new strike price (must be > 0).

        Returns
        -------
        BaseOption
            A new option instance with the updated strike, maintaining the same
            maturity and option type.

        Raises
        ------
        ValueError
            If `new_strike` is not positive.
        """
        if new_strike <= 0:
            raise ValueError("Strike price must be positive.")
        return replace(self, strike=new_strike)

    def with_maturity(self, new_maturity: float) -> "BaseOption":
        """
        Create a new option instance with an updated time to maturity.

        Parameters
        ----------
        new_maturity : float
            The new time to maturity in years (must be > 0).

        Returns
        -------
        BaseOption
            A new option instance with the updated maturity, maintaining the same
            strike and option type.

        Raises
        ------
        ValueError
            If `new_maturity` is not positive.
        """
        if new_maturity <= 0:
            raise ValueError("Time to maturity must be positive.")
        return replace(self, maturity=new_maturity)

    @property
    def companion_option(self) -> "BaseOption":
        """
        Create a companion option with the same strike and maturity but with the
        option type inverted (i.e., call becomes put and vice versa).

        Returns
        -------
        BaseOption
            A new option instance with the `is_call` flag inverted.
        """
        return replace(self, is_call=not self.is_call)
