"""
Option
======

Implements an Option class that can represent a vanilla European or
American call/put on a single underlying.

TO DO: For exotic options, this class can be
subclass to introduce custom payoff logic.

Design & Usage
--------------
- The Option class inherits from BaseInstrument. It stores:
  * strike (float)
  * maturity (float, in years)
  * underlying_symbol (str)
  * option_type (str): 'call' or 'put'
  * option_style (str): 'european' or 'american' (default: 'european')
- The payoff(...) method calculates a standard vanilla payoff:
  max(spot - strike, 0) for calls, max(strike - spot, 0) for puts.
- The value(...) method raises NotImplementedError, enforcing that valuation 
  must be performed via a Model/Technique combination (e.g., Black-Scholes, 
  Binomial, Monte Carlo).

Performance & HPC
-----------------
- For HPC or large-scale pricing runs (e.g., across many strikes and maturities):
  * Vectorized payoff calculations with NumPy is used in a technique class.
  * Parallelization (multiprocessing/joblib) can speed up scenario or path-based 
    valuations.
- Thread-safety concerns, if any, would primarily be around mutable state updates 
  (e.g., dynamic parameters). In this basic class, the parameters are static, so
  concurrency issues are minimal.

Examples
--------
>>> from src.instruments.option import Option
>>> my_option = Option("AAPL", maturity=1.0, strike=100.0, option_type="call")
>>> payoff_value = my_option.payoff(spot_price=105.0)
>>> print(payoff_value)
5.0
"""

import threading
from typing import Optional, Any

from src.instruments.base_instrument import BaseInstrument

VALID_OPTION_TYPES = {"call", "put"}
VALID_OPTION_STYLES = {"european", "american"}


class Option(BaseInstrument):
    """
    A generic option instrument, which can represent either European or American
    style calls/puts on a single underlying asset.

    Notes
    -----
    - Valuation is intentionally left to external Model/Technique classes.
    - For exotic payoffs (e.g., Asian, Barrier), one can either extend this class
      with a custom payoff(...) or create a new class.

    Parameters
    ----------
    underlying_symbol : str
        The symbol of the underlying asset (e.g., 'AAPL').
    maturity : float
        Time to maturity in years.
    strike : float
        The strike (exercise) price of the option.
    option_type : str, {'call', 'put'}, default='call'
        Whether the option is a call or a put.
    option_style : str, {'european', 'american'}, default='european'
        The exercise style of the option.
    market_env : Any, optional
        An optional MarketEnvironment for fetching market data
        (spot price, interest rates, etc.).

    Attributes
    ----------
    _strike : float
        The strike price of the option.
    _option_type : str
        'call' or 'put'.
    _option_style : str
        'european' or 'american'.
    """

    def __init__(
        self,
        underlying_symbol: str,
        maturity: float,
        strike: float,
        option_type: str = "call",
        option_style: str = "european",
        market_env: Optional[Any] = None,
    ) -> None:
        super().__init__(
            underlying_symbol=underlying_symbol,
            maturity=maturity,
            market_env=market_env,
        )

        if strike < 0:
            raise ValueError("strike must be non-negative.")

        if option_type not in VALID_OPTION_TYPES:
            raise ValueError(
                f"option_type must be one of {VALID_OPTION_TYPES}, got {option_type}."
            )

        if option_style not in VALID_OPTION_STYLES:
            raise ValueError(
                f"option_style must be one of {VALID_OPTION_STYLES}, got {option_style}."
            )

        self._strike = strike
        self._option_type = option_type.lower()  # 'call' or 'put'
        self._option_style = option_style.lower()  # 'european' or 'american'

        self._lock = threading.Lock()

    @property
    def strike(self) -> float:
        """
        float : The strike price of the option.
        """
        return self._strike

    @property
    def option_type(self) -> str:
        """
        str : 'call' or 'put'.
        """
        return self._option_type

    @property
    def option_style(self) -> str:
        """
        str : 'european' or 'american'.
        """
        return self._option_style

    def payoff(self, spot_price: float, **kwargs) -> float:
        """
        Compute the payoff at expiration for a vanilla option.

        For a call option: payoff = max(spot_price - strike, 0)
        For a put option: payoff = max(strike - spot_price, 0)

        Parameters
        ----------
        spot_price : float
            The final price of the underlying at maturity.
        **kwargs :
            Additional arguments (unused by default).

        Returns
        -------
        float
            The payoff value.

        Notes
        -----
        - In more exotic or path-dependent cases (Asian, Barrier, etc.), override
          this method or create a specialized option class.
        """
        if self._option_type == "call":
            return max(spot_price - self._strike, 0.0)
        else:  # put
            return max(self._strike - spot_price, 0.0)

    def value(self, market_env: Any = None, **kwargs) -> float:
        """
        Retrieve the fair value of the option.
        Computed via a 'Technique'

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError(
            "Option does not implement an internal valuation. "
            "Use a pricing model and technique to compute its fair value."
        )

    def __repr__(self) -> str:
        """
        String representation for debugging.
        """
        with self._lock:
            info = (
                f"symbol={self.underlying_symbol}, "
                f"maturity={self.maturity}, "
                f"strike={self._strike}, "
                f"option_type={self._option_type}, "
                f"option_style={self._option_style}"
            )
        return f"Option({info})"
