"""
BaseInstrument
==============

This module defines the abstract foundation for all financial instruments
in the library. It provides a consistent interface for derived
instruments (e.g. options, stocks) to ensure interoperability
with different pricing models and techniques.

Notes
-----
- Inherits from abc.ABC to enforce abstract methods.
- Designed for seamless integration of financial instruments for the future.
- Provides a standard method to retrieve an instrument's fair value (`value`).
- Optionally holds a reference to a MarketEnvironment for market data.
- Provides a default `payoff` method that can be overridden for derivatives.
- HPC/real-time: concurrency control (e.g., locks) may be
    required if multiple threads share and modify the same MarketEnvironment.
"""

from src.market.market_environment import MarketEnvironment

import abc
from typing import Any, Optional


class BaseInstrument(abc.ABC):
    """
    Abstract base class for all financial instruments.

    Provides:
    ---------
    - Common attributes: underlying_symbol and maturity
    - Optional reference to a MarketEnvironment for contextual data
    - A consistent interface for derived instruments

    Implements:
    -----------
    - Abstract method `value(...)` that derived classes must override.
    - Default `payoff(...)` which raises NotImplementedError unless overridden.

    """

    def __init__(
        self,
        underlying_symbol: str,
        maturity: float,
        market_env: Optional[MarketEnvironment] = None,
    ) -> None:
        """
        Initialize a new financial instrument.

        Parameters
        ----------
        underlying_symbol : str
            A string identifying the underlying asset (e.g., 'AAPL').
        maturity : float
            Time to maturity in years. Must be a positive number.
        market_env : Optional[MarketEnvironment], default=None
            A MarketEnvironment providing market parameters.

        Raises
        ------
        ValueError
            If an invalid underlying_symbol or maturity is provided.
        """
        if not underlying_symbol or not isinstance(underlying_symbol, str):
            raise ValueError("underlying_symbol must be a non-empty string.")
        if maturity <= 0:
            raise ValueError("maturity must be a positive float (in years).")

        self._underlying_symbol = underlying_symbol
        self._maturity = maturity
        self._market_env = market_env

    @property
    def underlying_symbol(self) -> str:
        """
        str : The underlying asset's symbol.
        """
        return self._underlying_symbol

    @property
    def maturity(self) -> float:
        """
        float : The instrument's time to maturity in years.
        """
        return self._maturity

    @property
    def market_env(self) -> Optional[MarketEnvironment]:
        """
        Optional[MarketEnvironment] :
            The MarketEnvironment associated with this instrument, if any.
        """
        return self._market_env

    def get_underlying_identifier(self) -> str:
        """
        Retrieve the underlying identifier of this instrument.

        Returns
        -------
        str
            The underlying asset's symbol.
        """
        return self._underlying_symbol

    @abc.abstractmethod
    def value(self, market_env: Any = None, **kwargs) -> float:
        """
        Calculate the fair value of the instrument, given a market environment.

        Parameters
        ----------
        market_env : Any, optional
            The market data or environment used for valuation.
            If None, may default to self._market_env.
        **kwargs : dict, optional
            Additional parameters that might influence valuation.

        Returns
        -------
        float
            Fair value of the instrument.

        Notes
        -----
        - Derived classes must override this method to provide instrument-
          specific valuation logic.
        """
        pass

    def payoff(self, spot_price: float, **kwargs) -> float:
        """
        Compute payoff given the underlying spot price.

        This default method raises a NotImplementedError unless overridden
        in a derivative-like instrument.

        Parameters
        ----------
        spot_price : float
            Current or final price of the underlying asset.
        **kwargs : dict, optional
            Additional parameters for the payoff calculation.

        Returns
        -------
        float
            The instrument payoff.

        Raises
        ------
        NotImplementedError
            If the derived instrument does not define its own payoff method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not define a payoff method."
        )

    @property
    def init_info(self):
        """
        Returns a list of key initialization parameters and their values.

        Returns
        -------
        list of tuple
            A list of (param_name, param_value) pairs for debugging/logging.
        """
        return [
            ("underlying_symbol", self._underlying_symbol),
            ("maturity", self._maturity),
            ("market_env", self._market_env),
        ]

    def __repr__(self) -> str:
        """
        String representation for debugging.

        Returns
        -------
        str
            Class name and initialization parameters.
        """
        info_str = ", ".join(f"{key}={value!r}" for key, value in self.init_info)
        return f"{self.__class__.__name__}({info_str})"
