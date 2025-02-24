"""
test_integration_technique.py
=============================

This file contains a suite of pytest tests for the integration-based Fourier pricing
technique implemented in fourier_pricing_technique.py. In particular, it tests:
  - The static method integrate_phi for both call and put options.
  - The price and delta methods.
  - The implied_volatility method via monkey-patching the price method.
  - The vega method via monkey-patching the finite-difference routine.
  - The __repr__ method.

Note:
  To avoid numerical issues with oscillatory integrals, tests for the integration
  routines use S=1 and K=1 (so that log(K)=0). This yields analytically tractable values.
  
To run these tests:
    pytest test_integration_technique.py
"""

import math
import numpy as np
import pytest

from src.techniques.characteristic.integration_technique import IntegrationTechnique

# Import the technique class from the module under test.


# =============================================================================
# Dummy classes for dependencies
# =============================================================================
class FakeInstrument:
    """
    A fake option instrument with the minimal required attributes.
    """

    def __init__(self, strike: float, maturity: float, option_type: str):
        self.strike = strike
        self.maturity = maturity
        self.option_type = option_type  # "Call" or "Put"


class FakeStock:
    """
    A fake underlying asset with required attributes.
    """

    def __init__(self, spot: float, volatility: float, dividend: float):
        self.spot = spot
        self.volatility = volatility
        self.dividend = dividend


class FakeMarketEnvironment:
    """
    A fake market environment providing the risk-free rate.
    """

    def __init__(self, rate: float):
        self.rate = rate


class FakeModel:
    """
    A fake pricing model.

    For integration tests the characteristic function is defined to be trivial
    (ϕ(u)=1), so that the integration returns known values.

    For implied volatility and vega tests, the model carries a 'sigma' attribute.
    """

    def __init__(self, sigma: float):
        self.sigma = sigma

    def with_volatility(self, sigma: float) -> "FakeModel":
        return FakeModel(sigma)

    def characteristic_function(self, T: float, S: float, r: float, q: float):
        # For integration tests, return a trivial characteristic function.
        def phi(u: complex) -> complex:
            return 1

        return phi


# =============================================================================
# Helper functions for implied volatility and vega tests
# =============================================================================
def fake_price(instrument, underlying, model, market_env, **kwargs):
    """
    Fake price function that returns model.sigma. This creates a monotonic relationship,
    so that solving price - target_price = 0 yields an implied volatility equal to target_price.
    """
    return model.sigma


def fake_finite_diff_1st(func, base_args, param, step):
    """
    A simple central-difference finite difference approximation.
    """
    args_plus = base_args.copy()
    args_minus = base_args.copy()
    args_plus[param] += step
    args_minus[param] -= step
    return (func(**args_plus) - func(**args_minus)) / (2 * step)


# =============================================================================
# Tests for IntegrationTechnique (Fourier pricing technique)
# =============================================================================

# For the integration tests below we choose parameters that yield log(K)=0.
# (Set S=1 and K=1 so that the Fourier phase factors become unity.)


@pytest.fixture
def basic_params():
    # Use parameters that yield a zero log-strike.
    S = 1.0
    K = 1.0
    r = 0.05
    T = 1.0
    q = 0.02
    # Compute adjusted factors.
    adjS = S * math.exp(-q * T)
    adjK = K * math.exp(-r * T)
    return S, K, r, T, q, adjS, adjK


def test_integrate_phi_call(basic_params):
    """
    Test the static integrate_phi method for a call option.
    With ϕ(u)=1 and K=1 (so log(K)=0), the transformation functions become zero:
      trfPhi(u)= imag(exp(0)*1)/u = 0,
      trfTwi(u)= 0.
    Hence, the integrals yield A = B = 0,
      pITMCall = 0.5 and deltaCall = 0.5.
    Then, for a call:
      price = adjS*0.5 - adjK*0.5 and delta = 0.5.
    """
    S, K, r, T, q, adjS, adjK = basic_params

    phi = lambda u: 1
    price, delta = IntegrationTechnique.integrate_phi(phi, S, K, r, T, q, call=True)

    expected_pITM = 0.5  # 0.5 + 0/π
    expected_delta = 0.5  # 0.5 + 0/π
    expected_price = adjS * expected_delta - adjK * expected_pITM

    assert math.isclose(
        price, expected_price, rel_tol=1e-6
    ), f"Call price expected {expected_price}, got {price}"
    assert math.isclose(
        delta, expected_delta, rel_tol=1e-6
    ), f"Call delta expected {expected_delta}, got {delta}"


def test_integrate_phi_put(basic_params):
    """
    Test the integrate_phi method for a put option.
    With K=1 and ϕ(u)=1, as above we have pITMCall = deltaCall = 0.5.
    For a put:
      price = adjK*(1 - 0.5) - adjS*(1 - 0.5) = 0.5*(adjK - adjS),
      delta = 0.5 - 1 = -0.5.
    """
    S, K, r, T, q, adjS, adjK = basic_params

    phi = lambda u: 1
    price, delta = IntegrationTechnique.integrate_phi(phi, S, K, r, T, q, call=False)

    expected_pITM = 0.5
    expected_delta = expected_pITM - 1  # -0.5
    expected_price = adjK * (1 - expected_pITM) - adjS * (1 - expected_pITM)

    assert math.isclose(
        price, expected_price, rel_tol=1e-6
    ), f"Put price expected {expected_price}, got {price}"
    assert math.isclose(
        delta, expected_delta, rel_tol=1e-6
    ), f"Put delta expected {expected_delta}, got {delta}"


def test_price_call(basic_params):
    """
    Test the price method for a call option.
    With our trivial characteristic function and K=1, the integration yields:
      pITMCall = 0.5 and delta = 0.5.
    Thus, the call price becomes:
      price = adjS*0.5 - adjK*0.5.
    """
    S, K, r, T, q, adjS, adjK = basic_params

    technique = IntegrationTechnique(cache_results=False)
    instrument = FakeInstrument(strike=K, maturity=T, option_type="Call")
    # Use S=1 here.
    underlying = FakeStock(spot=S, volatility=0.2, dividend=q)
    market_env = FakeMarketEnvironment(rate=r)
    model = FakeModel(sigma=0.2)

    expected_price = adjS * 0.5 - adjK * 0.5

    price = technique.price(instrument, underlying, model, market_env)
    assert math.isclose(
        price, expected_price, rel_tol=1e-6
    ), f"Call option price expected {expected_price}, got {price}"


def test_price_put(basic_params):
    """
    Test the price method for a put option.
    With the trivial ϕ(u)=1 and K=1, the put price is:
      price = adjK*(1-0.5) - adjS*(1-0.5) = 0.5*(adjK - adjS).
    """
    S, K, r, T, q, adjS, adjK = basic_params

    technique = IntegrationTechnique(cache_results=False)
    instrument = FakeInstrument(strike=K, maturity=T, option_type="Put")
    underlying = FakeStock(spot=S, volatility=0.2, dividend=q)
    market_env = FakeMarketEnvironment(rate=r)
    model = FakeModel(sigma=0.2)

    expected_price = adjK * 0.5 - adjS * 0.5

    price = technique.price(instrument, underlying, model, market_env)
    assert math.isclose(
        price, expected_price, rel_tol=1e-6
    ), f"Put option price expected {expected_price}, got {price}"


def test_delta_call(basic_params):
    """
    Test the delta method for a call option.
    For K=1 and ϕ(u)=1, the integration yields a delta of 0.5.
    """
    S, K, r, T, q, _, _ = basic_params

    technique = IntegrationTechnique(cache_results=False)
    instrument = FakeInstrument(strike=K, maturity=T, option_type="Call")
    underlying = FakeStock(spot=S, volatility=0.2, dividend=q)
    market_env = FakeMarketEnvironment(rate=r)
    model = FakeModel(sigma=0.2)

    expected_delta = 0.5
    delta = technique.delta(instrument, underlying, model, market_env)
    assert math.isclose(
        delta, expected_delta, rel_tol=1e-6
    ), f"Call delta expected {expected_delta}, got {delta}"


def test_delta_put(basic_params):
    """
    Test the delta method for a put option.
    With the trivial integration result, put delta = delta_call - 1 = 0.5 - 1 = -0.5.
    """
    S, K, r, T, q, _, _ = basic_params

    technique = IntegrationTechnique(cache_results=False)
    instrument = FakeInstrument(strike=K, maturity=T, option_type="Put")
    underlying = FakeStock(spot=S, volatility=0.2, dividend=q)
    market_env = FakeMarketEnvironment(rate=r)
    model = FakeModel(sigma=0.2)

    expected_delta = 0.5 - 1  # -0.5
    delta = technique.delta(instrument, underlying, model, market_env)
    assert math.isclose(
        delta, expected_delta, rel_tol=1e-6
    ), f"Put delta expected {expected_delta}, got {delta}"


def test_implied_volatility():
    """
    Test the implied_volatility method by monkey-patching the technique's price method.
    Here we override price to return model.sigma, so that setting a target_price forces the
    implied volatility to be equal to target_price.
    """
    technique = IntegrationTechnique(cache_results=False)
    technique.price = fake_price  # Override with our fake monotonic pricing function.

    instrument = FakeInstrument(strike=1, maturity=1.0, option_type="Call")
    underlying = FakeStock(spot=1, volatility=0.2, dividend=0.02)
    market_env = FakeMarketEnvironment(rate=0.05)
    model = FakeModel(sigma=0.2)

    target_price = (
        0.25  # With fake_price, we expect implied volatility to equal target_price.
    )
    implied_vol = technique.implied_volatility(
        instrument, underlying, model, market_env, target_price=target_price
    )
    assert math.isclose(
        implied_vol, target_price, rel_tol=1e-6
    ), f"Implied volatility expected {target_price}, got {implied_vol}"


def test_vega():
    """
    Test the vega method by monkey-patching the finite-difference routine and the pricing function.
    With fake_price returning model.sigma (a linear function), the derivative with respect to sigma
    is 1.
    """
    technique = IntegrationTechnique(cache_results=False)
    technique._finite_diff_1st = (
        fake_finite_diff_1st  # Override finite-difference method.
    )
    technique.get_price = fake_price  # Override get_price to use fake_price.

    instrument = FakeInstrument(strike=1, maturity=1.0, option_type="Call")
    underlying = FakeStock(spot=1, volatility=0.2, dividend=0.02)
    market_env = FakeMarketEnvironment(rate=0.05)
    model = FakeModel(sigma=0.2)

    expected_vega = 1.0
    vega_value = technique.vega(instrument, underlying, model, market_env)
    assert math.isclose(
        vega_value, expected_vega, rel_tol=1e-6
    ), f"Vega expected {expected_vega}, got {vega_value}"


def test_repr():
    """
    Test that the __repr__ method returns a string containing the class name and cache_results flag.
    """
    technique = IntegrationTechnique(cache_results=True)
    rep = repr(technique)
    assert (
        "IntegrationTechnique" in rep or "Fourier" in rep
    ), f"__repr__ output unexpected: {rep}"
    assert (
        "cache_results=True" in rep
    ), f"__repr__ output does not include cache_results flag: {rep}"
