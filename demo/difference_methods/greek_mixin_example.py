"""
finite_diff_greek_example.py
============================

Demonstrates the usage of the GreekMixin for computing option Greeks via
finite differences. The dummy price function is defined as:

    price = underlying.spot^2 + underlying.volatility^3 + instrument.maturity^2 + market_env.rate

which has analytical derivatives:
    Delta = 2 * underlying.spot
    Gamma = 2
    Vega  = 3 * underlying.volatility^2
    Theta = -2 * instrument.maturity
    Rho   = 1

The example prints the computed Greeks along with their expected values.
"""


# Dummy classes for demonstration purposes
class DummyInstrument:
    def __init__(self, maturity: float):
        self.maturity = maturity


class DummyUnderlying:
    def __init__(self, spot: float, volatility: float):
        self.spot = spot
        self.volatility = volatility


class DummyMarketEnv:
    def __init__(self, rate: float):
        self.rate = rate


from src.mixins.greek_mixin import GreekMixin


class DummyTechnique(GreekMixin):
    """
    Dummy technique that implements a simple price function:

        price = underlying.spot^2 + underlying.volatility^3 + instrument.maturity^2 + market_env.rate

    Analytical derivatives:
      - Delta = 2 * underlying.spot
      - Gamma = 2
      - Vega  = 3 * underlying.volatility^2
      - Theta = -2 * instrument.maturity
      - Rho   = 1
    """

    def __init__(self, parallel: bool = False):
        super().__init__(parallel=parallel)

    def price(self, instrument, underlying, model, market_env, **kwargs) -> float:
        return (
            underlying.spot**2
            + underlying.volatility**3
            + instrument.maturity**2
            + market_env.rate
        )

    def implied_volatility(
        self, instrument, underlying, model, market_env, target_price: float
    ) -> float:
        # Dummy implementation for compatibility
        return 0.0

    def graph(self, instrument, underlying, model, market_env) -> None:
        # Dummy implementation: no graphing in this example.
        pass


def example_finite_diff_greeks():
    # Define sample parameters
    spot = 100.0
    volatility = 0.3
    maturity = 2.0
    rate = 0.05

    # Create dummy instrument, underlying, and market environment.
    instrument = DummyInstrument(maturity=maturity)
    underlying = DummyUnderlying(spot=spot, volatility=volatility)
    market_env = DummyMarketEnv(rate=rate)
    # Dummy model (not used in this example)
    model = None

    # Initialize the dummy technique (using serial finite differences)
    technique = DummyTechnique(parallel=False)

    # Compute the option price (for demonstration)
    price_val = technique.price(instrument, underlying, model, market_env)

    # Compute Greeks via finite differences
    delta_val = technique.delta(instrument, underlying, model, market_env)
    gamma_val = technique.gamma(instrument, underlying, model, market_env)
    vega_val = technique.vega(instrument, underlying, model, market_env)
    theta_val = technique.theta(instrument, underlying, model, market_env)
    rho_val = technique.rho(instrument, underlying, model, market_env)

    # Expected analytical values:
    expected_delta = 2 * spot  # 2 * 100 = 200
    expected_gamma = 2  # Second derivative of S^2 is 2
    expected_vega = 3 * (volatility**2)  # 3 * (0.3^2) = 0.27
    expected_theta = -2 * maturity  # -2 * 2.0 = 4.0
    expected_rho = 1  # Derivative with respect to rate is 1

    # Print the results
    print("Finite Difference Greeks Example")
    print("----------------------------------")
    print(f"Option Price: {price_val:.4f}\n")
    print("Computed Greeks:")
    print(f"  Delta: {delta_val:.4f}   (Expected: {expected_delta:.4f})")
    print(f"  Gamma: {gamma_val:.4f}   (Expected: {expected_gamma:.4f})")
    print(f"  Vega:  {vega_val:.4f}   (Expected: {expected_vega:.4f})")
    print(f"  Theta: {theta_val:.4f}   (Expected: {expected_theta:.4f})")
    print(f"  Rho:   {rho_val:.4f}   (Expected: {expected_rho:.4f})")


if __name__ == "__main__":
    example_finite_diff_greeks()
