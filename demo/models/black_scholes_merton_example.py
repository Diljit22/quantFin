"""
example_black_scholes_merton.py
================================

Demonstration of using the BlackScholesMerton model.

This example creates an instance of BlackScholesMerton with a given volatility,
validates its parameters, obtains the characteristic function, and computes its
value at a sample complex argument.
"""

from src.models.black_scholes_merton import BlackScholesMerton


def example_black_scholes_merton() -> None:
    """
    Demonstrate usage of the BlackScholesMerton model.

    Creates a BlackScholesMerton instance, obtains its characteristic function,
    and computes the function's value for a sample complex argument.

    Returns
    -------
    None
    """
    # Define model and market parameters.
    sigma = 0.2  # Volatility (20%)
    spot = 100.0  # Current spot price
    t = 1.0  # Time to maturity in years
    r = 0.05  # Risk-free interest rate (5%)
    q = 0.02  # Continuous dividend yield (2%)

    # Create a BlackScholesMerton model instance.
    model = BlackScholesMerton(sigma)

    # Validate model parameters.
    model.validate_params()

    # Obtain the characteristic function for the specified market parameters.
    phi = model.characteristic_function(t, spot, r, q)

    # Define a sample complex argument.
    u = 1 + 2j

    # Compute the characteristic function value at u.
    phi_value = phi(u)

    print("\n=== BlackScholesMerton Model Example ===\n")
    print("Model parameters:")
    print(f"  Volatility (sigma):      {sigma}")
    print(f"  Spot price (S):          {spot}")
    print(f"  Time to maturity (T):    {t}")
    print(f"  Risk-free rate (r):      {r}")
    print(f"  Dividend yield (q):      {q}\n")
    print("Sample complex argument:")
    print(f"  u = {u}\n")
    print("Characteristic function value:")
    print(f"  phi(u) = {phi_value}\n")
    print("Done with BlackScholesMerton example.\n")


if __name__ == "__main__":
    example_black_scholes_merton()
