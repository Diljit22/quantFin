"""
base_technique.py
=================

Defines the abstract foundation for all pricing techniques in the quant library.
A Technique is responsible for numerically computing the price, Greeks, and
implied volatility of a given instrument under a specified model and market
environment.

Examples of techniques include:
- Closed-form solutions (e.g. Black-Scholes)
- Lattice methods (e.g. Binomial)
- Monte Carlo
- FFT-based approaches

This base class only specifies the interface. Concrete implementations
should handle performance optimizations, vectorization, concurrency,
and caching as needed.


Extensibility
-------------
- To add a new technique, inherit from `BaseTechnique` and implement:
  * price(...)

  The following can be implemented by inheriting from FiniteDiffTechnique
  * implied_volatility(...)
  * delta(...)
  * gamma(...)
  * vega(...)
  * theta(...)
  * rho(...)
- A default `graph(...)` method is provided by `GraphMixin`, which this class inherits.
  You can override it if you need a custom plotting approach.
"""

import abc
from typing import Any, Dict, Optional

from src.mixins.graph_mixin import GraphMixin


class BaseTechnique(abc.ABC, GraphMixin):
    """
    Abstract base class for pricing techniques in a quantitative library.

    A Technique is responsible for:
      - Pricing a given instrument under a specified model & market environment.
      - Computing standard option Greeks individually (delta, gamma, vega, theta, rho).
      - Computing implied volatilities if needed.

    Child classes should:
      - Implement numeric optimizations or parallelization as needed.
      - Potentially use caching or memoization to speed up repeated calculations.
    """

    def __init__(self, cache_results: bool = False) -> None:
        """
        Initialize the base technique.

        Parameters
        ----------
        cache_results : bool, optional
            If True, caching is enabled for repeated calls with
            identical arguments (default=False).
        """
        self._cache_results = cache_results
        # A simple dictionary-based cache; child classes can
        # implement advanced caching if desired.
        self._cache: Dict[Any, Any] = {}

    @abc.abstractmethod
    def price(self, instrument: Any, underlying, model: Any, market_env: Any) -> float:
        """
        Compute the fair value (price) of the instrument.

        Parameters
        ----------
        instrument : Any
            The instrument (e.g., an option) to be priced.
        model : Any
            The model (e.g., Black-Scholes, Heston) with underlying assumptions.
        market_env : Any
            The market environment (e.g., risk-free rates, discount factors).

        Returns
        -------
        float
            The computed fair value (present value) of the instrument.
        """
        pass

    @abc.abstractmethod
    def implied_volatility(
        self,
        instrument: Any,
        underlying,
        model: Any,
        market_env: Any,
        target_price: float,
    ) -> float:
        """
        Compute the implied volatility that reproduces a target market price.

        Parameters
        ----------
        instrument : Any
            The instrument (e.g., option) whose implied vol is sought.
        model : Any
            The model used as a base (often BlackScholesMerton).
        market_env : Any
            The market environment data (e.g., risk-free rate).
        target_price : float
            The observed or market price to match via implied vol.

        Returns
        -------
        float
            The implied volatility (decimal) that matches the target_price
            under the chosen model & environment.
        """
        pass

    # ---- Individual Greeks: delta, gamma, vega, theta, rho ----
    @abc.abstractmethod
    def delta(self, instrument: Any, underlying, model: Any, market_env: Any) -> float:
        """
        Delta: dPrice / dSpot

        Parameters
        ----------
        instrument : Any
            The instrument for which we compute delta.
        model : Any
            The model providing dynamics (e.g., volatility).
        market_env : Any
            The market environment (e.g., interest rates).

        Returns
        -------
        float
            The delta of the instrument.
        """
        pass

    @abc.abstractmethod
    def gamma(self, instrument: Any, underlying, model: Any, market_env: Any) -> float:
        """
        Gamma: d^2Price / dSpot^2

        Returns
        -------
        float
            The gamma of the instrument.
        """
        pass

    @abc.abstractmethod
    def vega(self, instrument: Any, underlying, model: Any, market_env: Any) -> float:
        """
        Vega: dPrice / dVol

        Returns
        -------
        float
            The vega of the instrument (sensitivity to volatility).
        """
        pass

    @abc.abstractmethod
    def theta(self, instrument: Any, underlying, model: Any, market_env: Any) -> float:
        """
        Theta: dPrice / dTime (time decay)

        Returns
        -------
        float
            The theta of the instrument (sensitivity to time).
        """
        pass

    @abc.abstractmethod
    def rho(self, instrument: Any, underlying, model: Any, market_env: Any) -> float:
        """
        Rho: d Price / d Rate (sensitivity to interest rate)

        Returns
        -------
        float
            The rho of the instrument.
        """
        pass

    def _make_cache_key(
        self, instrument, underlying, model, market_env, **kwargs
    ) -> tuple:
        """
        Construct a stable, hashable key for caching calls to price(...), Greeks, etc.

        The default logic calls `.__hashable_state__()` on each object (if present). If
        an object does not define `.__hashable_state__()`, we fall back to using `id(obj)`
        which effectively disables caching across runs with different object
        references.

        Child classes may override this if they want custom logic.
        """

        def safe_hash_state(obj):
            """
            Try calling obj.__hashable_state__(), otherwise default to id(obj).
            Ideally, each class (instrument, underlying, model, market_env)
            defines to_tuple() to produce a purely hashable (float/bool/etc.) signature.
            """
            to_t = getattr(obj, "__hashable_state__", None)
            if callable(to_t):
                return to_t()
            else:
                return id(obj)

        # Base key for main objects
        base_key = (
            safe_hash_state(instrument),
            safe_hash_state(underlying),
            safe_hash_state(model),
            safe_hash_state(market_env),
        )

        # sorted by key name for consistent ordering
        if kwargs:
            sorted_kwargs = tuple(sorted(kwargs.items()))
            return (base_key, sorted_kwargs)
        else:
            return base_key

    def _lookup_cache(self, cache_key: tuple) -> Optional[Any]:
        """
        Retrieve a cached result if it exists, otherwise return None.
        """
        return self._cache.get(cache_key, None)

    def _store_cache(self, cache_key: tuple, result: Any) -> None:
        """
        Store a result in the cache with the given key.
        """
        self._cache[cache_key] = result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(cache_results={self._cache_results})"
