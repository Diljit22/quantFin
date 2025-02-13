# techniques/base_technique.py

"""
BaseTechnique
=============

This module defines the abstract foundation for all pricing techniques in the
quant library. A Technique is responsible for numerically computing the price,
Greeks, and implied volatility of a given instrument under a specified model
and market environment. Examples of techniques include closed-form solutions
(e.g. Black-Scholes), lattice methods (e.g. Binomial), finite-difference PDE
solvers, Monte Carlo, and FFT-based approaches.

Performance & HPC
-----------------
- This base class only specifies the interface. Concrete implementations
  (e.g., PDE or Monte Carlo) should handle performance optimizations:
  * Vectorization (NumPy) for batch computations
  * Parallelization (multiprocessing/joblib) for large grid expansions or
    many Monte Carlo paths
  * Thread-safety if MarketEnvironment or other shared data is updated 
    concurrently
- Caching or memoization could be beneficial if repeated pricing or Greek
  calculations are required.

Extensibility
-------------
- To add a new technique, simply inherit from BaseTechnique and implement
  all abstract methods (price, implied_volatility, and the Greeks).
- Graphing can be implemented optionally (the `graph` method raises
  NotImplementedError by default).

Examples
--------
A minimal example of implementing a closed-form Black-Scholes technique:

>>> class BlackScholesTechnique(BaseTechnique):
...     def price(self, instrument, model, market_env, **kwargs) -> float:
...         # Implement closed-form logic here
...         pass
...     def implied_volatility(self, instrument, model, market_env, market_price, **kwargs) -> float:
...         # Use a root-finder to invert the BS price
...         pass
...     def delta(self, instrument, model, market_env, **kwargs) -> float:
...         # Compute partial derivative w.r.t. spot
...         pass
...     # etc. for gamma, vega, theta, rho
"""

import abc
from typing import Any

from src.instruments.base_instrument import BaseInstrument
from src.models.base_model import BaseModel


class BaseTechnique(abc.ABC):
    """
    Abstract base class for all pricing methods and associated Greeks/implied vol.

    A PricingTechnique:
    -------------------
    - Consumes an Instrument, a Model, and a MarketEnvironment (or equivalent
      data), then returns a numeric result such as a price or Greeks.
    - May also compute the implied volatility given a market-observed price.
    - Different numerical approaches (closed-form, lattice, PDE, Monte Carlo, FFT)
      will implement these methods differently.

    Notes
    -----
    - HPC considerations:
      * Parallel or vectorized implementations are often used in derived classes.
      * Thread-safety must be ensured if the same technique object is called
        concurrently in a multi-threaded environment.
      * If many calculations are repeated (e.g., across strikes), caching may be
        implemented to avoid redundant computations.
    """

    @abc.abstractmethod
    def price(
        self, instrument: BaseInstrument, model: BaseModel, market_env: Any, **kwargs
    ) -> float:
        """
        Compute the fair value of the instrument under the given model
        and market environment.

        Parameters
        ----------
        instrument : BaseInstrument
            The financial instrument to be priced (e.g., a European option).
        model : BaseModel
            The model specifying how to interpret or simulate the underlying process.
        market_env : Any
            Market data or environment needed for pricing (spot price,
            interest rate, volatility surface, etc.).
        **kwargs : dict, optional
            Additional parameters for specialized usage or tuning (e.g.,
            numerical tolerances, grid sizes, random seeds).

        Returns
        -------
        float
            The fair value of the instrument according to this technique.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by the subclass.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def implied_volatility(
        self,
        instrument: BaseInstrument,
        model: BaseModel,
        market_env: Any,
        market_price: float,
        **kwargs,
    ) -> float:
        """
        Compute the implied volatility given an observed market price.

        Parameters
        ----------
        instrument : BaseInstrument
            The financial instrument (usually an option).
        model : BaseModel
            The model specifying how to interpret the underlying process.
        market_env : Any
            Market data needed for pricing (spot price, interest rate, etc.).
        market_price : float
            Observed market price of the instrument.
        **kwargs : dict, optional
            Additional parameters (e.g., tolerance levels, initial vol guess).

        Returns
        -------
        float
            The implied volatility consistent with the given market price.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by the subclass.
        """
        raise NotImplementedError

    # ----------------------------------------------------------------
    # Greeks
    # ----------------------------------------------------------------

    @abc.abstractmethod
    def delta(
        self, instrument: BaseInstrument, model: BaseModel, market_env: Any, **kwargs
    ) -> float:
        """
        Compute the Delta of the instrument under this technique.

        Delta is the partial derivative of price with respect to
        the underlying spot price.

        Parameters
        ----------
        instrument : BaseInstrument
            The instrument for which to compute Delta.
        model : BaseModel
            The pricing model context.
        market_env : Any
            Market data, including spot price, rates, etc.
        **kwargs : dict, optional
            Additional parameters (e.g., finite difference steps,
            numerical tolerances).

        Returns
        -------
        float
            The Delta of the instrument.

        Raises
        ------
        NotImplementedError
            If not implemented by the subclass.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def gamma(
        self, instrument: BaseInstrument, model: BaseModel, market_env: Any, **kwargs
    ) -> float:
        """
        Compute the Gamma of the instrument under this technique.

        Gamma is the second partial derivative of price with respect to
        the underlying spot price.

        Parameters
        ----------
        instrument : BaseInstrument
            The instrument for which to compute Gamma.
        model : BaseModel
            The pricing model context.
        market_env : Any
            Market data, including spot price, rates, etc.
        **kwargs : dict, optional
            Additional parameters for numerical approximation.

        Returns
        -------
        float
            The Gamma of the instrument.

        Raises
        ------
        NotImplementedError
            If not implemented by the subclass.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def vega(
        self, instrument: BaseInstrument, model: BaseModel, market_env: Any, **kwargs
    ) -> float:
        """
        Compute the Vega of the instrument under this technique.

        Vega is the partial derivative of price with respect to the volatility
        of the underlying.

        Parameters
        ----------
        instrument : BaseInstrument
            The instrument for which to compute Vega.
        model : BaseModel
            The pricing model context.
        market_env : Any
            Market data, including spot price, rates, volatility, etc.
        **kwargs : dict, optional
            Additional parameters for numerical approximation.

        Returns
        -------
        float
            The Vega of the instrument.

        Raises
        ------
        NotImplementedError
            If not implemented by the subclass.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def theta(
        self, instrument: BaseInstrument, model: BaseModel, market_env: Any, **kwargs
    ) -> float:
        """
        Compute the Theta of the instrument under this technique.

        Theta is the partial derivative of price with respect to
        time to maturity.

        Parameters
        ----------
        instrument : BaseInstrument
            The instrument for which to compute Theta.
        model : BaseModel
            The pricing model context.
        market_env : Any
            Market data, including spot price, rates, etc.
        **kwargs : dict, optional
            Additional parameters for numerical approximation.

        Returns
        -------
        float
            The Theta of the instrument.

        Raises
        ------
        NotImplementedError
            If not implemented by the subclass.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def rho(
        self, instrument: BaseInstrument, model: BaseModel, market_env: Any, **kwargs
    ) -> float:
        """
        Compute the Rho of the instrument under this technique.

        Rho is the partial derivative of price with respect to
        the interest rate.

        Parameters
        ----------
        instrument : BaseInstrument
            The instrument for which to compute Rho.
        model : BaseModel
            The pricing model context.
        market_env : Any
            Market data, including spot price, rates, etc.
        **kwargs : dict, optional
            Additional parameters for numerical approximation.

        Returns
        -------
        float
            The Rho of the instrument.

        Raises
        ------
        NotImplementedError
            If not implemented by the subclass.
        """
        raise NotImplementedError

    # ----------------------------------------------------------------
    # (Optional) Graphing
    # ----------------------------------------------------------------

    def graph(
        self,
        instrument: BaseInstrument,
        model: BaseModel,
        graph_type: str,
        market_env: Any,
        **kwargs,
    ) -> None:
        """
        Generate a graph for the specified metric or dimension
        (e.g., option payoff, Greek vs. strike, etc.).

        This default implementation raises NotImplementedError.
        Subclasses may override to support specialized visualizations.

        Parameters
        ----------
        instrument : BaseInstrument
            The financial instrument to visualize.
        model : BaseModel
            The pricing model context.
        graph_type : str
            The type of graph to generate (e.g., "payoff", "greeks_vs_strike").
        market_env : Any
            Market data or environment for the visualization (e.g.
            range of spot prices or interest rates).
        **kwargs : dict, optional
            Additional plotting parameters or configuration.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement graphing.
        """
        raise NotImplementedError(
            f"Graphing not implemented for {self.__class__.__name__}"
        )
