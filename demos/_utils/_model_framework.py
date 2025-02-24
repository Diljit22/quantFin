from datetime import datetime
from src.financial_calculations.parity_bounds import put_call_parity
import csv
import os


class OptionExampleRunner:
    def __init__(self, model, technique, underlying, instrument, market_env):
        self.model = model
        self.technique = technique
        self.underlying = underlying
        self.instrument = instrument
        self.market_env = market_env

    def run_silent(self):
        """
        Run pricing and return results as a dictionary instead of printing.

        Returns
        -------
        dict
            Contains keys: ['price', 'delta', 'gamma', 'vega', 'theta', 'rho',
                            'parity_comp', 'implied_vol']
        """
        # 1) Compute the option price
        price = self.technique.price(
            self.instrument, self.underlying, self.model, self.market_env
        )

        # 2) Compute Greeks
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

        # 3) Compute put-call parity
        if self.instrument.is_call:
            parity_comp = put_call_parity(
                price,
                self.underlying.spot,
                self.instrument.strike,
                self.market_env.rate,
                self.instrument.maturity,
                self.underlying.dividend,
                price_call=True,
            )
        else:
            parity_comp = put_call_parity(
                price,
                self.underlying.spot,
                self.instrument.strike,
                self.market_env.rate,
                self.instrument.maturity,
                self.underlying.dividend,
                price_call=False,
            )

        # 4) Compute implied volatility
        iv = self.technique.implied_volatility(
            self.instrument,
            self.underlying,
            self.model,
            self.market_env,
            target_price=price,
        )

        return {
            "price": price,
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "theta": theta,
            "rho": rho,
            "parity_comp": parity_comp,
            "implied_vol": iv,
        }

    def run(self):
        """
        (Original) Run the pricing example, print all details, and then call self.save_to_csv().
        This is useful for standalone demonstrations, but not for side-by-side comparisons.
        """
        print("\n=== Option Pricing Example ===\n")
        print("Model:", self.model.__class__.__name__)
        print("Technique:", self.technique.__class__.__name__)
        print("Underlying:", self.underlying)
        print("Instrument:", self.instrument)
        print("Market Environment:", self.market_env)
        print("\n----------------------------\n")

        # Get results from run_silent
        results = self.run_silent()

        # Print them
        print("Option Price: {:.4f}".format(results["price"]))

        print("\nGreeks:")
        print("  Delta: {:.4f}".format(results["delta"]))
        print("  Gamma: {:.4f}".format(results["gamma"]))
        print("  Vega:  {:.4f}".format(results["vega"]))
        print("  Theta: {:.4f}".format(results["theta"]))
        print("  Rho:   {:.4f}".format(results["rho"]))

        if self.instrument.is_call:
            print("\nPut–Call Parity:")
            print(
                "  Given Call Price, computed Put Price: {:.4f}".format(
                    results["parity_comp"]
                )
            )
        else:
            print("\nPut–Call Parity:")
            print(
                "  Given Put Price, computed Call Price: {:.4f}".format(
                    results["parity_comp"]
                )
            )

        print("\nImplied Volatility: {:.4f}".format(results["implied_vol"]))

        # Save to CSV
        self.save_to_csv(
            results["price"],
            results["delta"],
            results["gamma"],
            results["vega"],
            results["theta"],
            results["rho"],
            results["implied_vol"],
        )

    def save_to_csv(
        self, price, delta, gamma, vega, theta, rho, iv, filename="option_results.csv"
    ):
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

        # Ensure the artifacts folder exists
        artifacts_folder = "artifacts"
        os.makedirs(artifacts_folder, exist_ok=True)

        # Construct the full file path in the artifacts folder
        full_path = os.path.join(artifacts_folder, filename)

        try:
            with open(full_path, "a", newline="") as f:
                writer = csv.writer(f)
                # Write header if file is empty
                if f.tell() == 0:
                    writer.writerow(header)
                writer.writerow(row)
            print("\nResults saved to", full_path)
        except Exception as e:
            print("Error saving results to CSV:", e)
