"""
test_finite_diff_technique.py
=============================

Tests the FiniteDifferenceTechnique (via a dummy subclass) for:
  - Price calculation.
  - Finite-difference Greeks: delta, gamma, vega, theta, and rho.
  - Implied volatility using a bracket-based solver.

The dummy price function is defined as:
    price = underlying.spot^2 + underlying.volatility^3 + instrument.maturity^2 + market_env.rate

Analytical derivatives:
    Delta = 2 * underlying.spot
    Gamma = 2
    Vega  = 3 * underlying.volatility^2
    Theta = 2 * instrument.maturity
    Rho   = 1
"""

import math
import pytest
from scipy.optimize import brentq


# --- Dummy Domain Classes ---
class DummyInstrument:
    def __init__(
        self, maturity: float, strike: float = 100.0, option_type: str = "Call"
    ):
        self.maturity = maturity
        self.strike = strike
        self.option_type = option_type


class DummyUnderlying:
    def __init__(self, spot: float, volatility: float, dividend: float = 0.0):
        self.spot = spot
        self.volatility = volatility
        self.dividend = dividend


class DummyMarketEnv:
    def __init__(self, rate: float):
        self.rate = rate


# --- Dummy Finite Difference Technique Subclass ---
from src.techniques.finite_diff_technique import FiniteDifferenceTechnique


class DummyFiniteDiffTechnique(FiniteDifferenceTechnique):
    """
    A dummy subclass of FiniteDifferenceTechnique that implements .price(...)
    using the dummy formula:

        price = underlying.spot^2 + underlying.volatility^3 + instrument.maturity^2 + market_env.rate
    """

    def price(self, instrument, underlying, model, market_env, **kwargs) -> float:
        return (
            underlying.spot**2
            + underlying.volatility**3
            + instrument.maturity**2
            + market_env.rate
        )

    def implied_volatility(
        self, instrument, underlying, model, market_env, target_price: float, **kwargs
    ) -> float:
        # Build a unique cache key based on relevant parameters.
        spot = getattr(underlying, "spot", None)
        strike = getattr(instrument, "strike", None)
        maturity = getattr(instrument, "maturity", None)
        opt_type = getattr(instrument, "option_type", None)
        cache_key = (spot, strike, maturity, opt_type, target_price)
        if cache_key in self._iv_cache:
            return self._iv_cache[cache_key]

        if maturity is None or maturity <= 1e-14:
            raise ValueError(
                "Cannot compute IV for an almost expired (or invalid) option."
            )
        if target_price < 0:
            raise ValueError("Market price cannot be negative for IV calculation.")

        # Bracket volatility in [1e-9, 5.0]
        low_vol, high_vol = 1e-9, 5.0
        tol = kwargs.get("tol", 1e-7)
        max_iter = kwargs.get("max_iter", 200)
        initial_guess = kwargs.get("initial_guess", 0.2)

        def price_diff(sigma_val: float) -> float:
            original_sigma = getattr(underlying, "volatility", None)
            if original_sigma is None and hasattr(model, "sigma"):
                original_sigma = model.sigma
            self._set_vol(underlying, model, sigma_val)
            p = self.price(instrument, underlying, model, market_env)
            self._set_vol(underlying, model, original_sigma)
            return p - target_price

        f_low = price_diff(low_vol)
        f_high = price_diff(high_vol)

        if f_low * f_high < 0.0:
            try:
                iv_est = brentq(
                    price_diff, low_vol, high_vol, xtol=tol, maxiter=max_iter
                )
            except Exception:
                iv_est = self._secant_iv(price_diff, initial_guess, tol, max_iter)
        else:
            iv_est = self._secant_iv(price_diff, initial_guess, tol, max_iter)

        self._iv_cache[cache_key] = iv_est
        return iv_est

    def graph(self, instrument, underlying, model, market_env) -> None:
        # No graphing functionality in this subclass.
        pass


# --- Pytest Fixture ---
@pytest.fixture
def dummy_data():
    """
    Creates dummy instrument, underlying, market environment, a dummy model (None),
    and an instance of DummyFiniteDiffTechnique.
    """
    spot = 100.0
    volatility = 0.3
    maturity = 2.0
    rate = 0.05
    instrument = DummyInstrument(maturity=maturity)
    underlying = DummyUnderlying(spot=spot, volatility=volatility)
    market_env = DummyMarketEnv(rate=rate)
    model = None  # Not used in dummy price function.
    # Disable caching for price computations (to avoid conflicts with mutable objects),
    # but use local _iv_cache for implied volatility.
    technique = DummyFiniteDiffTechnique(cache_results=False, parallel=False)
    return instrument, underlying, market_env, model, technique


# --- Test Cases ---
def test_price(dummy_data):
    instrument, underlying, market_env, model, technique = dummy_data
    price_val = technique.price(instrument, underlying, model, market_env)
    expected_price = (100.0**2) + (0.3**3) + (2.0**2) + 0.05
    assert math.isclose(
        price_val, expected_price, rel_tol=1e-5
    ), f"Price: {price_val} != {expected_price}"


def test_delta(dummy_data):
    instrument, underlying, market_env, model, technique = dummy_data
    delta_val = technique.delta(instrument, underlying, model, market_env)
    expected_delta = 2 * underlying.spot  # 2 * 100 = 200
    assert math.isclose(
        delta_val, expected_delta, rel_tol=1e-5
    ), f"Delta: {delta_val} != {expected_delta}"


def test_gamma(dummy_data):
    instrument, underlying, market_env, model, technique = dummy_data
    gamma_val = technique.gamma(instrument, underlying, model, market_env)
    expected_gamma = 2  # Second derivative of S^2 is 2.
    assert math.isclose(
        gamma_val, expected_gamma, rel_tol=1e-5
    ), f"Gamma: {gamma_val} != {expected_gamma}"


def test_vega(dummy_data):
    instrument, underlying, market_env, model, technique = dummy_data
    vega_val = technique.vega(instrument, underlying, model, market_env)
    expected_vega = 3 * (underlying.volatility**2)  # 3 * (0.3^2) = 0.27
    assert math.isclose(
        vega_val, expected_vega, rel_tol=1e-5
    ), f"Vega: {vega_val} != {expected_vega}"


def test_theta(dummy_data):
    instrument, underlying, market_env, model, technique = dummy_data
    theta_val = technique.theta(instrument, underlying, model, market_env)
    expected_theta = -2 * instrument.maturity  # -2 * 2.0 = 4.0
    assert math.isclose(
        theta_val, expected_theta, rel_tol=1e-5
    ), f"Theta: {-theta_val} != {expected_theta}"


def test_rho(dummy_data):
    instrument, underlying, market_env, model, technique = dummy_data
    rho_val = technique.rho(instrument, underlying, model, market_env)
    expected_rho = 1  # Derivative with respect to rate is 1.
    assert math.isclose(
        rho_val, expected_rho, rel_tol=1e-5
    ), f"Rho: {rho_val} != {expected_rho}"


def test_implied_volatility(dummy_data):
    instrument, underlying, market_env, model, technique = dummy_data
    # Compute the target price using the dummy price function with current volatility (0.3).
    target_price = technique.price(instrument, underlying, model, market_env)
    # The implied volatility should be close to the current underlying volatility (0.3).
    iv = technique.implied_volatility(
        instrument, underlying, model, market_env, target_price
    )
    assert math.isclose(
        iv, underlying.volatility, rel_tol=1e-5
    ), f"IV: {iv} != {underlying.volatility}"

    # Test with a modified target price corresponding to a different volatility.
    desired_vol = 0.4
    # New price = spot^2 + (desired_vol^3) + maturity^2 + rate
    new_price = (
        (underlying.spot**2)
        + (desired_vol**3)
        + (instrument.maturity**2)
        + market_env.rate
    )
    iv_new = technique.implied_volatility(
        instrument, underlying, model, market_env, new_price
    )
    assert math.isclose(
        iv_new, desired_vol, rel_tol=1e-5
    ), f"IV: {iv_new} != {desired_vol}"
