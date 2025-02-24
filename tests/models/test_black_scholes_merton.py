# test_black_scholes_merton.py

import math
import cmath
import pytest
from src.models.black_scholes_merton import BlackScholesMerton


def test_valid_instantiation():
    """Test that a BlackScholesMerton model is instantiated with the correct sigma."""
    bsm = BlackScholesMerton(sigma=0.2)
    params = bsm.get_params()
    assert params["sigma"] == 0.2
    # Also check that the model name is set correctly.
    assert "BlackScholesMerton" in bsm.model_name


def test_negative_sigma_raises_value_error():
    """Test that initializing with a negative sigma raises a ValueError."""
    with pytest.raises(ValueError):
        BlackScholesMerton(sigma=-0.1)


def test_characteristic_function_phi_at_zero():
    """
    Test that the characteristic function returns 1 at u=0.
    [Follows from E[e^(0)] = 1]
    """
    bsm = BlackScholesMerton(sigma=0.2)
    t = 1.0
    spot = 100.0
    r = 0.05
    q = 0.02
    phi = bsm.characteristic_function(t, spot, r, q)
    result = phi(0)
    # For any valid characteristic function, φ(0) should equal 1.
    assert abs(result - 1) < 1e-12


def test_characteristic_function_value():
    """
    Test that the characteristic function returns the expected value for u = 1.

    For Black-Scholes, given:
      drift = ln(spot) + (r - q - 0.5 * sigma^2)*t
      half_var = 0.5 * sigma^2
    the characteristic function is:
      exp( i u * drift - half_var * t * u^2 )
    """
    sigma = 0.2
    t = 1.0
    spot = 100.0
    r = 0.05
    q = 0.02
    bsm = BlackScholesMerton(sigma=sigma)
    phi = bsm.characteristic_function(t, spot, r, q)

    u = 1.0
    half_var = 0.5 * sigma * sigma  # 0.02
    drift = math.log(spot) + (r - q - half_var) * t  # ln(100) + (0.05 - 0.02 - 0.02)
    expected = cmath.exp(1j * u * drift - half_var * t * (u**2))
    result = phi(u)
    # Compare real and imaginary parts separately with a tight tolerance.
    assert math.isclose(result.real, expected.real, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(result.imag, expected.imag, rel_tol=1e-9, abs_tol=1e-9)


def test_repr_contains_model_name_and_sigma():
    """Test that the __repr__ output contains the model name and sigma value."""
    sigma = 0.2
    bsm = BlackScholesMerton(sigma=sigma)
    rep = repr(bsm)
    assert "BlackScholesMerton" in rep
    # Check that sigma is represented (formatted as 0.2 or with more decimals)
    assert str(sigma) in rep or f"{sigma:.4f}" in rep
