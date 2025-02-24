"""
test_greek_mixin.py
===================

Pytest module for testing the GreekMixin class.
It defines a dummy technique class that implements a simple price function with known derivatives.
The tests verify that the finite-difference approximations for delta, gamma, vega, theta, and rho
match the expected analytical derivatives.

price = S^2 + sigma^3 + T^2 + r
delta = d(price)/dS = 2*S
gamma = d(delta)/dS = 2
vega = d(price)/d(sigma) = 3*sigma^2
theta = -d(price)/dT = -2*T
rho = d(price)/dr = 1
"""

import math
import pytest
from src.mixins.greek_mixin import GreekMixin

# ------------------------------------------------------------------------------
# Dummy Classes for Testing
# ------------------------------------------------------------------------------


class DummyInstrument:
    def __init__(self, maturity: float):
        self.maturity = maturity

    def __hash__(self):
        return hash(self.maturity)

    def __eq__(self, other):
        return self.maturity == other.maturity


class DummyUnderlying:
    def __init__(self, spot: float, volatility: float):
        self.spot = spot
        self.volatility = volatility

    def __hash__(self):
        return hash((self.spot, self.volatility))

    def __eq__(self, other):
        return (self.spot, self.volatility) == (other.spot, other.volatility)


class DummyMarketEnv:
    def __init__(self, rate: float):
        self.rate = rate

    def __hash__(self):
        return hash(self.rate)

    def __eq__(self, other):
        return self.rate == other.rate


class DummyModel:
    def __hash__(self):
        return 0

    def __eq__(self, other):
        return True


# ------------------------------------------------------------------------------
# DummyTechnique: Inherits from GreekMixin and implements .price(...)
# ------------------------------------------------------------------------------


class DummyTechnique(GreekMixin):
    """
    Dummy technique that implements a simple price function:

        price = underlying.spot^2 + underlying.volatility^3 + instrument.maturity^2 + market_env.rate

    which has known derivatives:
      - Delta   = 2 * underlying.spot
      - Gamma   = 2
      - Vega    = 3 * underlying.volatility^2
      - Theta   = 2 * instrument.maturity
      - Rho     = 1
    """

    def __init__(self, parallel: bool = False):
        super().__init__(parallel)

    def price(
        self, instrument: any, underlying: any, model: any, market_env: any, **kwargs
    ) -> float:
        return (
            underlying.spot**2
            + underlying.volatility**3
            + instrument.maturity**2
            + market_env.rate
        )


# Fixtures


@pytest.fixture
def dummy_objects():
    """
    Creates dummy instrument, underlying, market_env, and model for testing.
    """
    instrument = DummyInstrument(maturity=2.0)
    underlying = DummyUnderlying(spot=100.0, volatility=0.3)
    market_env = DummyMarketEnv(rate=0.05)
    model = DummyModel()
    return instrument, underlying, market_env, model


# Serial Finite Difference (parallel=False)


def test_delta_serial(dummy_objects):
    instrument, underlying, market_env, model = dummy_objects
    technique = DummyTechnique(parallel=False)
    delta_val = technique.delta(instrument, underlying, model, market_env)
    expected_delta = 2 * underlying.spot  # 2 * 100 = 200
    assert math.isclose(
        delta_val, expected_delta, rel_tol=1e-4
    ), f"Delta (serial): {delta_val} != {expected_delta}"


def test_gamma_serial(dummy_objects):
    instrument, underlying, market_env, model = dummy_objects
    technique = DummyTechnique(parallel=False)
    gamma_val = technique.gamma(instrument, underlying, model, market_env)
    expected_gamma = 2  # second derivative of S^2
    assert math.isclose(
        gamma_val, expected_gamma, rel_tol=1e-4
    ), f"Gamma (serial): {gamma_val} != {expected_gamma}"


def test_vega_serial(dummy_objects):
    instrument, underlying, market_env, model = dummy_objects
    technique = DummyTechnique(parallel=False)
    vega_val = technique.vega(instrument, underlying, model, market_env)
    expected_vega = 3 * (underlying.volatility**2)  # 3 * (0.3^2) = 0.27
    assert math.isclose(
        vega_val, expected_vega, rel_tol=1e-4
    ), f"Vega (serial): {vega_val} != {expected_vega}"


def test_theta_serial(dummy_objects):
    instrument, underlying, market_env, model = dummy_objects
    technique = DummyTechnique(parallel=False)
    theta_val = technique.theta(instrument, underlying, model, market_env)
    expected_theta = -2 * instrument.maturity  # -2 * 2.0 = -4.0
    assert math.isclose(
        theta_val, expected_theta, rel_tol=1e-4
    ), f"Theta (serial): {theta_val} != {expected_theta}"


def test_rho_serial(dummy_objects):
    instrument, underlying, market_env, model = dummy_objects
    technique = DummyTechnique(parallel=False)
    rho_val = technique.rho(instrument, underlying, model, market_env)
    expected_rho = 1  # derivative of r is 1
    assert math.isclose(
        rho_val, expected_rho, rel_tol=1e-4
    ), f"Rho (serial): {rho_val} != {expected_rho}"


# Parallel Finite Difference (parallel=True)


def test_delta_parallel(dummy_objects):
    instrument, underlying, market_env, model = dummy_objects
    technique = DummyTechnique(parallel=True)
    delta_val = technique.delta(instrument, underlying, model, market_env)
    expected_delta = 2 * underlying.spot
    assert math.isclose(
        delta_val, expected_delta, rel_tol=1e-4
    ), f"Delta (parallel): {delta_val} != {expected_delta}"


def test_gamma_parallel(dummy_objects):
    instrument, underlying, market_env, model = dummy_objects
    technique = DummyTechnique(parallel=True)
    gamma_val = technique.gamma(instrument, underlying, model, market_env)
    expected_gamma = 2
    assert math.isclose(
        gamma_val, expected_gamma, rel_tol=1e-4
    ), f"Gamma (parallel): {gamma_val} != {expected_gamma}"


def test_vega_parallel(dummy_objects):
    instrument, underlying, market_env, model = dummy_objects
    technique = DummyTechnique(parallel=True)
    vega_val = technique.vega(instrument, underlying, model, market_env)
    expected_vega = 3 * (underlying.volatility**2)
    assert math.isclose(
        vega_val, expected_vega, rel_tol=1e-4
    ), f"Vega (parallel): {vega_val} != {expected_vega}"


def test_theta_parallel(dummy_objects):
    instrument, underlying, market_env, model = dummy_objects
    technique = DummyTechnique(parallel=True)
    theta_val = technique.theta(instrument, underlying, model, market_env)
    expected_theta = -2 * instrument.maturity
    assert math.isclose(
        theta_val, expected_theta, rel_tol=1e-4
    ), f"Theta (parallel): {theta_val} != {expected_theta}"


def test_rho_parallel(dummy_objects):
    instrument, underlying, market_env, model = dummy_objects
    technique = DummyTechnique(parallel=True)
    rho_val = technique.rho(instrument, underlying, model, market_env)
    expected_rho = 1
    assert math.isclose(
        rho_val, expected_rho, rel_tol=1e-4
    ), f"Rho (parallel): {rho_val} != {expected_rho}"
