"""
_pricing_enviroments.py
=================

Creates three sample European call options and their equivalent put options.
These options are used for demonstration and testing.

Parameters Used:
    Underlying:
        Spot:       100.00
        Volatility: 20.00%
        Dividend:   2.00%
        Risk-Free:  5.00%
S
Option Set 1:
    Strike:     90.00
    Maturity:   1.00 years

Option Set 2:
    Strike:     100.00
    Maturity:   0.75 years

Option Set 3:
    Strike:     110.00
    Maturity:   0.50 years
"""

from src.underlyings.stock import Stock
from src.market.market_environment import MarketEnvironment
from src.instruments.european_option import EuropeanOption


def create_pricing_enviroments():
    """
    Create sample European call options and their equivalent put options.

    Returns
    -------
    underlying : Stock
        The underlying asset.
    market_env : MarketEnvironment
        The market environment.
    call_options : list of EuropeanOption
        A list of call options.
    put_options : list of EuropeanOption
        A list of put options corresponding to the call options.
    """
    # Define common underlying and market parameters.
    spot_price = 100.0
    volatility = 0.20
    dividend = 0.02
    risk_free_rate = 0.05

    underlying = Stock(spot=spot_price, volatility=volatility, dividend=dividend)
    market_env = MarketEnvironment(rate=risk_free_rate)

    # Define three sets of option parameters.
    params = [
        {"strike": 90.0, "maturity": 1.0},
        {"strike": 100.0, "maturity": 0.75},
        {"strike": 110.0, "maturity": 0.5},
    ]

    call_options = []
    put_options = []

    for p in params:
        call_options.append(
            EuropeanOption(strike=p["strike"], maturity=p["maturity"], is_call=True)
        )
        put_options.append(
            EuropeanOption(strike=p["strike"], maturity=p["maturity"], is_call=False)
        )

    return underlying, market_env, call_options, put_options


def print_pricing_enviroments() -> None:
    """
    Print details for the sample options.
    """
    underlying, market_env, call_options, put_options = create_pricing_enviroments()

    print("Underlying:")
    print(underlying)
    print("\nMarket Environment:")
    print(market_env)

    for idx, (call, put) in enumerate(zip(call_options, put_options), start=1):
        print(f"\nOption Set {idx}:")
        print(f"  Call Option: {call}")
        print(f"  Put Option:  {put}")


if __name__ == "__main__":
    print_pricing_enviroments()
