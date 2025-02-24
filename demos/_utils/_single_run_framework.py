"""
example_framework.py
====================

A framework for running option pricing examples.
"""

import csv
from datetime import datetime
from src.financial_calculations.parity_bounds import put_call_parity


class OptionExampleRunner:
    def __init__(self, model, technique, underlying, instrument, market_env):
        """
        Initialize the example runner.

        Parameters
        ----------
        model : BaseModel
            The pricing model instance.
        technique : BaseTechnique
            The technique instance used for pricing and computing Greeks.
        underlying : Stock
            The underlying asset.
        instrument : BaseOption
            The option instrument.
        market_env : MarketEnvironment
            The market environment.
        """
        self.model = model
        self.technique = technique
        self.underlying = underlying
        self.instrument = instrument
        self.market_env = market_env

    def run(self):
        """Run the pricing example and print all details."""
        print("\n=== Option Pricing Example ===\n")
        print("Model:", self.model.__class__.__name__)
        print("Technique:", self.technique.__class__.__name__)
        print("Underlying:", self.underlying)
        print("Instrument:", self.instrument)
        print("Market Environment:", self.market_env)
        print("\n----------------------------\n")

        # Compute the option price.
        price = self.technique.price(
            self.instrument, self.underlying, self.model, self.market_env
        )
        print("Option Price: {:.4f}".format(price))

        # Compute Greeks.
        delta = self.technique.delta(
            self.instrument, self.underlying, self.model, self.market_env
        )
        gamma = self.technique.gamma(
            self.instrument, self.underlying, self.model, self.market_env
        )
        vega = self.technique.vega(
            self.instrument, self.underlying, self.model, self.market_env
        )
        theta = self.technique.theta(
            self.instrument, self.underlying, self.model, self.market_env
        )
        rho = self.technique.rho(
            self.instrument, self.underlying, self.model, self.market_env
        )

        print("\nGreeks:")
        print("  Delta: {:.4f}".format(delta))
        print("  Gamma: {:.4f}".format(gamma))
        print("  Vega:  {:.4f}".format(vega))
        print("  Theta: {:.4f}".format(theta))
        print("  Rho:   {:.4f}".format(rho))

        # Compute put-call parity:
        if self.instrument.is_call:
            computed_put = put_call_parity(
                price,
                self.underlying.spot,
                self.instrument.strike,
                self.market_env.rate,
                self.instrument.maturity,
                self.underlying.dividend,
                price_call=True,
            )
            print("\nPut–Call Parity:")
            print("  Given Call Price, computed Put Price: {:.4f}".format(computed_put))
        else:
            computed_call = put_call_parity(
                price,
                self.underlying.spot,
                self.instrument.strike,
                self.market_env.rate,
                self.instrument.maturity,
                self.underlying.dividend,
                price_call=False,
            )
            print("\nPut–Call Parity:")
            print(
                "  Given Put Price, computed Call Price: {:.4f}".format(computed_call)
            )

        # Compute implied volatility.
        iv = self.technique.implied_volatility(
            self.instrument,
            self.underlying,
            self.model,
            self.market_env,
            target_price=price,
        )
        print("\nImplied Volatility: {:.4f}".format(iv))

        # Save results to CSV.
        self.save_to_csv(price, delta, gamma, vega, theta, rho, iv)

    def save_to_csv(
        self, price, delta, gamma, vega, theta, rho, iv, filename="option_results.csv"
    ):
        """Save the computed results to a CSV file for easy viewing."""
        header = [
            "Timestamp",
            "Model",
            "Technique",
            "Underlying_Spot",
            "Strike",
            "Maturity",
            "Volatility",
            "Dividend",
            "RiskFreeRate",
            "OptionPrice",
            "Delta",
            "Gamma",
            "Vega",
            "Theta",
            "Rho",
            "ImpliedVol",
        ]
        row = [
            datetime.now().isoformat(),
            self.model.__class__.__name__,
            self.technique.__class__.__name__,
            self.underlying.spot,
            self.instrument.strike,
            self.instrument.maturity,
            self.underlying.volatility,
            self.underlying.dividend,
            self.market_env.rate,
            price,
            delta,
            gamma,
            vega,
            theta,
            rho,
            iv,
        ]
        try:
            with open(filename, "a", newline="") as f:
                writer = csv.writer(f)
                # Write header if file is empty.
                if f.tell() == 0:
                    writer.writerow(header)
                writer.writerow(row)
            print("\nResults saved to", filename)
        except Exception as e:
            print("Error saving results to CSV:", e)
