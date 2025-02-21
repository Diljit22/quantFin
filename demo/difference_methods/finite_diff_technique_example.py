"""
finite_diff_technique_example.py
================================

Demonstrates the usage of FiniteDifferenceTechnique for pricing an option
and computing Greeks via finite differences. The dummy price function is:

    price = underlying.spot^2 + underlying.volatility^3 + instrument.maturity^2 + market_env.rate

which has analytical derivatives:
    Delta = 2 * underlying.spot
    Gamma = 2
    Vega  = 3 * underlying.volatility^2
    Theta = -2 * instrument.maturity
    Rho   = 1

The example prints the computed Greeks along with their expected values.
"""


# --- Dummy Classes for Demonstration ---
class DummyInstrument:
    def __init__(self, maturity: float, strike: float, option_type: str = "Call"):
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


# --- FiniteDifferenceTechnique Import & Subclass ---
from src.techniques.finite_diff_technique import FiniteDifferenceTechnique


class DummyFiniteDifferenceTechnique(FiniteDifferenceTechnique):
    """
    A dummy subclass of FiniteDifferenceTechnique that implements the
    abstract .price(...) method using the dummy price function:

        price = underlying.spot^2 + underlying.volatility^3 + instrument.maturity^2 + market_env.rate
    """

    def price(self, instrument, underlying, model, market_env, **kwargs) -> float:
        return (
            underlying.spot**2
            + underlying.volatility**3
            + instrument.maturity**2
            + market_env.rate
        )

    def graph(self, instrument, underlying, model, market_env) -> None:
        # No graphing in this example.
        pass


# --- Example Usage ---
def example_finite_diff_technique():
    # Define sample parameters.
    spot = 100.0
    volatility = 0.3
    maturity = 2.0
    rate = 0.05
    strike = 100.0
    option_type = "Call"

    # Create dummy instrument, underlying, market environment.
    instrument = DummyInstrument(
        maturity=maturity, strike=strike, option_type=option_type
    )
    underlying = DummyUnderlying(spot=spot, volatility=volatility)
    market_env = DummyMarketEnv(rate=rate)
    model = None  # Not used in this dummy example.

    # Initialize the technique (using serial finite differences).
    technique = DummyFiniteDifferenceTechnique(parallel=False)

    # Compute the option price (for demonstration).
    price_val = technique.price(instrument, underlying, model, market_env)

    # Compute Greeks via finite differences.
    delta_val = technique.delta(instrument, underlying, model, market_env)
    gamma_val = technique.gamma(instrument, underlying, model, market_env)
    vega_val = technique.vega(instrument, underlying, model, market_env)
    theta_val = technique.theta(instrument, underlying, model, market_env)
    rho_val = technique.rho(instrument, underlying, model, market_env)

    iv_val = technique.implied_volatility(
        instrument, underlying, model, market_env, price_val
    )
    # Expected analytical values:
    expected_delta = 2 * spot  # 2 * 100 = 200
    expected_gamma = 2  # Second derivative of S^2 is 2
    expected_vega = 3 * (volatility**2)  # 3 * (0.3^2) = 0.27
    expected_theta = -2 * maturity  # -2 * 2.0 = 4.0
    expected_rho = 1  # Derivative with respect to rate is 1
    expected_iv = underlying.volatility

    # Print the results.
    print("Finite Difference Technique Example")
    print("-------------------------------------")
    print(f"Option Price: {price_val:.4f}\n")
    print("Computed Greeks:")
    print(f"  Delta: {delta_val:.4f}   (Expected: {expected_delta:.4f})")
    print(f"  Gamma: {gamma_val:.4f}   (Expected: {expected_gamma:.4f})")
    print(f"  Vega:  {vega_val:.4f}   (Expected: {expected_vega:.4f})")
    print(f"  Theta: {theta_val:.4f}   (Expected: {expected_theta:.4f})")
    print(f"  Rho:   {rho_val:.4f}   (Expected: {expected_rho:.4f})")
    print(f"  IV:   {iv_val:.4f}   (Expected: {expected_iv:.4f})")


if __name__ == "__main__":
    example_finite_diff_technique()
