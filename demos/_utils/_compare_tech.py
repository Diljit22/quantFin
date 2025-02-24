"""
compare_techniques.py
=====================

Demonstrates how to compare multiple techniques side-by-side for the
same model, underlying, and list of EuropeanOptions.

We do the following:
1) Load the environment from _pricing_environments.py
2) Use a single model (e.g., BlackScholesMerton)
3) Try multiple techniques on each of the 6 options (3 strikes, each Call+Put)
4) Print side-by-side results in a single table
"""

from tabulate import tabulate  # pip install tabulate for pretty tables

# Your local imports (adjust the paths as needed)
from demos._utils._pricing_environments import create_pricing_enviroments
from demos._utils._model_framework import OptionExampleRunner


def compare_techniques(techniques, model):
    """
    Runs the same model across multiple 'techniques' for each of the
    6 options (Call/Put sets).

    Parameters
    ----------
    techniques : list
        List of technique instances (e.g. [BlackScholesMertonTechnique(), ...])
    model : BaseModel
        The model (e.g. BlackScholesMerton instance) to be used for pricing.
    """
    # 1) Create the shared environment
    underlying, market_env, call_options, put_options = create_pricing_enviroments()

    # We'll combine the call and put options for iteration
    all_options = call_options + put_options  # 3 calls + 3 puts => 6 total

    # 2) We'll build a list of rows to feed into tabulate
    table_rows = []
    table_header = [
        "Option (Strike, Mat, Type)",
        "Technique",
        "Price",
        "Delta",
        "Gamma",
        "Vega",
        "Theta",
        "Rho",
        "ParityComp",
        "ImpVol",
    ]

    # 3) Evaluate each option with each technique, storing results
    for opt in all_options:
        for tech in techniques:
            runner = OptionExampleRunner(
                model=model,
                technique=tech,
                underlying=underlying,
                instrument=opt,
                market_env=market_env,
            )
            results = runner.run_silent()  # returns dict of metrics

            # Create a short descriptor for the option
            desc_str = f"({opt.strike:.1f}, {opt.maturity:.2f}, {'Call' if opt.is_call else 'Put'})"

            # Build one row for the table
            row = [
                desc_str,
                tech.__class__.__name__,
                f"{results['price']:.4f}",
                f"{results['delta']:.4f}",
                f"{results['gamma']:.4f}",
                f"{results['vega']:.4f}",
                f"{results['theta']:.4f}",
                f"{results['rho']:.4f}",
                f"{results['parity_comp']:.4f}",
                f"{results['implied_vol']:.4f}",
            ]
            table_rows.append(row)
        table_rows.append("-" * 120)

    # 4) Print a nice table of all results
    print("\n=== Techniques Comparison Table ===\n")
    print("Underlying:", underlying)
    print("Market Environment:", market_env)
    print("Number of Options:", len(all_options))
    print()
    # Print the aggregated side-by-side table
    print(tabulate(table_rows, headers=table_header, tablefmt="pretty"))
    print()
