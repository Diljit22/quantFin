"""
test_greek_mixin_lru_cache.py
=============================

A test module to verify that the `@lru_cache` in GreekMixin
actually prevents multiple evaluations of `price(...)` for
the same arguments.

"CountingTechnique" that inherits GreekMixin and
increments a counter every time `price(...)` is invoked.
"""

import pytest
from src.mixins.greek_mixin import GreekMixin

# Dummy classes for instrument, underlying, market_env, model


class SpyInstrument:
    def __init__(self, maturity: float):
        self.maturity = maturity

    def __hash__(self):
        return hash(self.maturity)

    def __eq__(self, other):
        return self.maturity == other.maturity


class SpyUnderlying:
    def __init__(self, spot: float, volatility: float):
        self.spot = spot
        self.volatility = volatility

    def __hash__(self):
        return hash((self.spot, self.volatility))

    def __eq__(self, other):
        return (self.spot, self.volatility) == (other.spot, other.volatility)


class SpyMarketEnv:
    def __init__(self, rate: float):
        self.rate = rate

    def __hash__(self):
        return hash(self.rate)

    def __eq__(self, other):
        return self.rate == other.rate


class SpyModel:
    def __hash__(self):
        return 0

    def __eq__(self, other):
        return True


# CountingTechnique: increments self.call_count every time real price calc is done


class CountingTechnique(GreekMixin):
    """
    Inherits from GreekMixin, so it has the @lru_cache'd `_cached_price(...)`.
    Every call to `price(...)` increments `call_count`.
    """

    def __init__(self, parallel: bool = False):
        super().__init__(parallel)
        self.call_count = 0

    def price(self, instrument, underlying, model, market_env, **kwargs) -> float:
        """
        price = (spot + volatility + maturity + rate).
        """
        self.call_count += 1
        return (
            underlying.spot
            + underlying.volatility
            + instrument.maturity
            + market_env.rate
        )


# Test LRU Cache


def test_lru_cache_avoids_repeated_price_calls():
    """
    1. Instantiate CountingTechnique (which increments .call_count on each call).
    2. Call price(...) multiple times with the SAME parameters.
    3. The second call should be served by the LRU cache, so .call_count won't increase.
    """

    technique = CountingTechnique(parallel=False)

    # Build dummy objects
    instrument = SpyInstrument(maturity=1.0)
    underlying = SpyUnderlying(spot=100.0, volatility=0.20)
    market_env = SpyMarketEnv(rate=0.05)
    model = SpyModel()

    # First call: should increment call_count from 0 to 1
    price1 = technique.price(instrument, underlying, model, market_env)
    assert (
        technique.call_count == 1
    ), f"First call_count should be 1, got {technique.call_count}"

    # Second call with identical parameters => from LRU cache => call_count stays 1
    price2 = technique.price(instrument, underlying, model, market_env)
    assert price2 == price1, "Prices for identical params should match."
    assert (
        technique.call_count == 1
    ), f"Because of LRU cache, call_count should still be 1. Got {technique.call_count}"


def test_lru_cache_with_different_params():
    """
    Show that calls with *different* parameters skip the LRU cache,
    so .call_count increments for the new set of parameters.
    """
    technique = CountingTechnique(parallel=False)

    instr1 = SpyInstrument(maturity=1.0)
    under1 = SpyUnderlying(spot=100.0, volatility=0.20)
    env1 = SpyMarketEnv(rate=0.05)
    model = SpyModel()

    instr2 = SpyInstrument(maturity=2.0)  # changed parameter
    under2 = SpyUnderlying(spot=100.0, volatility=0.20)
    env2 = SpyMarketEnv(rate=0.05)

    # First call: increments call_count to 1
    _ = technique.price(instr1, under1, model, env1)
    assert technique.call_count == 1

    # Second call (identical to first) => no increment
    _ = technique.price(instr1, under1, model, env1)
    assert technique.call_count == 1

    # Third call (diff param: maturity=2.0) => call_count increments again
    _ = technique.price(instr2, under2, model, env2)
    assert (
        technique.call_count == 2
    ), f"Changed maturity => new LRU cache entry => increment to 2. Got {technique.call_count}"
