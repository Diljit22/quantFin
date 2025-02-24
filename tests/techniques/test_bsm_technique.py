"""
test_bsm_technique.py
=====================

Pytest module for testing the BlackScholesMertonTechnique in bsm_technique.py.
It verifies that:
  - European call and put prices are computed correctly.
  - Greeks (delta, gamma, vega, theta, rho) match the expected values.
  - Implied volatility recovers the underlying volatility when using the computed price.
  - The implied volatility cache is populated and used.
  - Invalid inputs raise the expected errors.
  - The graph method (inherited from a mixin) produces a plot without error.
"""

import math
import pytest

from src.techniques.closed_forms.bsm_technique import (
    BlackScholesMertonTechnique,
    bs_call_price,
    bs_put_price,
    call_delta,
    put_delta,
    bs_gamma,
    bs_vega,
    call_theta,
    put_theta,
    call_rho,
    put_rho,
)
from src.underlyings.stock import Stock
from src.market.market_environment import MarketEnvironment
from src.instruments.european_option import EuropeanOption
from src.models.black_scholes_merton import BlackScholesMerton

# Fixtures for required objects


@pytest.fixture
def underlying():
    """
    Returns a Stock instance representing the underlying asset.
    """
    return Stock(spot=100.0, volatility=0.2, dividend=0.02)


@pytest.fixture
def market_env():
    """
    Returns a MarketEnvironment instance.
    """
    return MarketEnvironment(rate=0.05)


@pytest.fixture
def model():
    """
    Returns a BlackScholesMerton model instance.
    """
    return BlackScholesMerton(sigma=0.2)


@pytest.fixture
def call_option():
    """
    Returns a EuropeanOption representing a call option.
    """
    return EuropeanOption(strike=100.0, maturity=1.0, is_call=True)


@pytest.fixture
def put_option():
    """
    Returns a EuropeanOption representing a put option.
    """
    return EuropeanOption(strike=100.0, maturity=1.0, is_call=False)


@pytest.fixture
def technique():
    """
    Returns an instance of BlackScholesMertonTechnique with caching enabled.
    """
    return BlackScholesMertonTechnique(cache_results=True)


# Tests for pricing


def test_price_call(call_option, underlying, market_env, model, technique):
    """
    Test that the call price computed by the technique matches the
    closed-form Black-Scholes call price.
    """
    expected_price = bs_call_price(
        underlying.spot,
        call_option.strike,
        call_option.maturity,
        market_env.rate,
        underlying.dividend,
        underlying.volatility,
    )
    computed_price = technique.price(call_option, underlying, model, market_env)
    assert math.isclose(
        computed_price, expected_price, rel_tol=1e-5
    ), f"Computed call price {computed_price} does not match expected {expected_price}"


def test_price_put(put_option, underlying, market_env, model, technique):
    """
    Test that the put price computed by the technique matches the
    closed-form Black-Scholes put price.
    """
    expected_price = bs_put_price(
        underlying.spot,
        put_option.strike,
        put_option.maturity,
        market_env.rate,
        underlying.dividend,
        underlying.volatility,
    )
    computed_price = technique.price(put_option, underlying, model, market_env)
    assert math.isclose(
        computed_price, expected_price, rel_tol=1e-5
    ), f"Computed put price {computed_price} does not match expected {expected_price}"


# Tests for Greeks


def test_delta_call(call_option, underlying, market_env, model, technique):
    expected_delta = call_delta(
        underlying.spot,
        call_option.strike,
        call_option.maturity,
        market_env.rate,
        underlying.dividend,
        underlying.volatility,
    )
    computed_delta = technique.delta(call_option, underlying, model, market_env)
    assert math.isclose(
        computed_delta, expected_delta, rel_tol=1e-5
    ), f"Computed call delta {computed_delta} does not match expected {expected_delta}"


def test_delta_put(put_option, underlying, market_env, model, technique):
    expected_delta = put_delta(
        underlying.spot,
        put_option.strike,
        put_option.maturity,
        market_env.rate,
        underlying.dividend,
        underlying.volatility,
    )
    computed_delta = technique.delta(put_option, underlying, model, market_env)
    assert math.isclose(
        computed_delta, expected_delta, rel_tol=1e-5
    ), f"Computed put delta {computed_delta} does not match expected {expected_delta}"


def test_gamma(call_option, underlying, market_env, model, technique):
    expected_gamma = bs_gamma(
        underlying.spot,
        call_option.strike,
        call_option.maturity,
        market_env.rate,
        underlying.dividend,
        underlying.volatility,
    )
    computed_gamma = technique.gamma(call_option, underlying, model, market_env)
    assert math.isclose(
        computed_gamma, expected_gamma, rel_tol=1e-5
    ), f"Computed gamma {computed_gamma} does not match expected {expected_gamma}"


def test_vega(call_option, underlying, market_env, model, technique):
    expected_vega = bs_vega(
        underlying.spot,
        call_option.strike,
        call_option.maturity,
        market_env.rate,
        underlying.dividend,
        underlying.volatility,
    )
    computed_vega = technique.vega(call_option, underlying, model, market_env)
    assert math.isclose(
        computed_vega, expected_vega, rel_tol=1e-5
    ), f"Computed vega {computed_vega} does not match expected {expected_vega}"


def test_theta_call(call_option, underlying, market_env, model, technique):
    expected_theta = call_theta(
        underlying.spot,
        call_option.strike,
        call_option.maturity,
        market_env.rate,
        underlying.dividend,
        underlying.volatility,
    )
    computed_theta = technique.theta(call_option, underlying, model, market_env)
    assert math.isclose(
        computed_theta, expected_theta, rel_tol=1e-5
    ), f"Computed call theta {computed_theta} does not match expected {expected_theta}"


def test_theta_put(put_option, underlying, market_env, model, technique):
    expected_theta = put_theta(
        underlying.spot,
        put_option.strike,
        put_option.maturity,
        market_env.rate,
        underlying.dividend,
        underlying.volatility,
    )
    computed_theta = technique.theta(put_option, underlying, model, market_env)
    assert math.isclose(
        computed_theta, expected_theta, rel_tol=1e-5
    ), f"Computed put theta {computed_theta} does not match expected {expected_theta}"


def test_rho_call(call_option, underlying, market_env, model, technique):
    expected_rho = call_rho(
        underlying.spot,
        call_option.strike,
        call_option.maturity,
        market_env.rate,
        underlying.dividend,
        underlying.volatility,
    )
    computed_rho = technique.rho(call_option, underlying, model, market_env)
    assert math.isclose(
        computed_rho, expected_rho, rel_tol=1e-5
    ), f"Computed call rho {computed_rho} does not match expected {expected_rho}"


def test_rho_put(put_option, underlying, market_env, model, technique):
    expected_rho = put_rho(
        underlying.spot,
        put_option.strike,
        put_option.maturity,
        market_env.rate,
        underlying.dividend,
        underlying.volatility,
    )
    computed_rho = technique.rho(put_option, underlying, model, market_env)
    assert math.isclose(
        computed_rho, expected_rho, rel_tol=1e-5
    ), f"Computed put rho {computed_rho} does not match expected {expected_rho}"


# Tests for implied volatility and caching


def test_implied_volatility_call(call_option, underlying, market_env, model, technique):
    """
    For a call option priced using BSM, solving for IV with the computed price
    should recover the underlying volatility.
    """
    call_price = technique.price(call_option, underlying, model, market_env)
    iv = technique.implied_volatility(
        call_option, underlying, model, market_env, target_price=call_price
    )
    assert math.isclose(
        iv, underlying.volatility, rel_tol=1e-3
    ), f"Implied volatility {iv} does not match underlying volatility {underlying.volatility}"


def test_implied_volatility_put(put_option, underlying, market_env, model, technique):
    put_price = technique.price(put_option, underlying, model, market_env)
    iv = technique.implied_volatility(
        put_option, underlying, model, market_env, target_price=put_price
    )
    assert math.isclose(
        iv, underlying.volatility, rel_tol=1e-3
    ), f"Implied volatility {iv} does not match underlying volatility {underlying.volatility}"


def test_iv_cache(call_option, underlying, market_env, model, technique):
    """
    Ensure that the implied volatility cache is populated and that repeated
    calls with the same parameters return the cached value.
    """
    call_price = technique.price(call_option, underlying, model, market_env)
    # Clear the cache to start fresh.
    technique._iv_cache.clear()
    iv1 = technique.implied_volatility(
        call_option, underlying, model, market_env, target_price=call_price
    )
    cache_key = (
        underlying.spot,
        call_option.strike,
        call_option.maturity,
        call_option.option_type,
        call_price,
    )
    assert cache_key in technique._iv_cache, "Cache key not found after computing IV."
    cached_iv = technique._iv_cache[cache_key]
    iv2 = technique.implied_volatility(
        call_option, underlying, model, market_env, target_price=call_price
    )
    assert math.isclose(
        iv1, iv2, rel_tol=1e-10
    ), "Repeated IV calculations do not match."
    assert math.isclose(
        cached_iv, iv1, rel_tol=1e-10
    ), "Cached IV does not match computed IV."


def test_implied_volatility_invalid_target(
    call_option, underlying, market_env, model, technique
):
    """
    Test that a negative target price raises a ValueError in implied_volatility.
    """
    with pytest.raises(ValueError):
        technique.implied_volatility(
            call_option, underlying, model, market_env, target_price=-10
        )


def test_implied_volatility_near_expiry(
    call_option, underlying, market_env, model, technique
):
    """
    Test that attempting to compute IV for an almost expired option raises ValueError.
    """
    near_expiry_option = EuropeanOption(
        strike=call_option.strike, maturity=1e-15, is_call=True
    )
    with pytest.raises(ValueError):
        technique.implied_volatility(
            near_expiry_option, underlying, model, market_env, target_price=1.0
        )


# Test for the graphing method (provided via a graph_mixin)


def test_graph_method(
    call_option, underlying, market_env, model, technique, monkeypatch
):
    """
    Test that the graph method produces a plot without error.
    Monkey-patch plt.show to avoid blocking during tests.
    """
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)

    try:
        technique.graph(call_option, underlying, model, market_env)
    except Exception as e:
        pytest.fail(f"Graph method raised an exception: {e}")
