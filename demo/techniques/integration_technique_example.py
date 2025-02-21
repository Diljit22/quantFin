"""

==========================

Demonstrates the usage of the BlackScholesMertonTechnique for pricing
European call and put options as well as computing their Greeks and implied
volatilities.
"""

from src.techniques.characteristic.integration_technique import IntegrationTechnique
from src.underlyings.stock import Stock
from src.market.market_environment import MarketEnvironment
from src.instruments.european_option import EuropeanOption
from src.models.black_scholes_merton import BlackScholesMerton
from src.financial_calculations.parity_bounds import put_call_parity


def example_int():
    # Define sample market and option parameters
    spot_price = 100.0
    strike = 100.0
    maturity = 1.0
    volatility = 0.20
    dividend = 0.02
    risk_free_rate = 0.05

    # Create the underlying asset, market environment, and model objects
    underlying = Stock(spot=spot_price, volatility=volatility, dividend=dividend)
    market_env = MarketEnvironment(rate=risk_free_rate)
    model = BlackScholesMerton(sigma=volatility)

    # Create a call option and a put option instrument
    call_option = EuropeanOption(strike=strike, maturity=maturity, is_call=True)
    put_option = EuropeanOption(strike=strike, maturity=maturity, is_call=False)

    # Initialize the Black-Scholes-Merton technique with caching enabled
    technique = IntegrationTechnique()

    # Compute option prices
    call_price = technique.price(call_option, underlying, model, market_env)
    put_price = technique.price(put_option, underlying, model, market_env)

    print("European Option Pricing & Greeks")
    print("--------------------------------")
    print(f"Underlying Spot:       {spot_price:.2f}")
    print(f"Strike:                {strike:.2f}")
    print(f"Maturity:              {maturity:.2f} years")
    print(f"Volatility:            {volatility:.2%}")
    print(f"Dividend Yield:        {dividend:.2%}")
    print(f"Risk-Free Rate:        {risk_free_rate:.2%}\n")

    print(f"Call Option Price:     {call_price:.4f}")
    print(f"Put Option Price:      {put_price:.4f}\n")

    computed_put = put_call_parity(
        call_price,
        spot_price,
        strike,
        risk_free_rate,
        maturity,
        dividend,
        price_call=True,
    )
    computed_call = put_call_parity(
        put_price,
        spot_price,
        strike,
        risk_free_rate,
        maturity,
        dividend,
        price_call=False,
    )
    # computed_r = implied_rate(call_price, put_price, spot_price, strike, maturity, dividend, eps=1e-6, max_iter=100)
    print(
        f"put_call_parity (Call Option Price is {call_price:.4f}): computed put price = {computed_put:.4f}"
    )
    print(
        f"put_call_parity (Put Option Price is {put_price:.4f}):  computed call price = {computed_call:.4f}\n"
    )
    # print(f"implied_rate: computed risk-free rate = {computed_r:.4f}\n")
    print("Done with parity examples.\n")

    # Compute Greeks for the call option
    call_delta = technique.delta(call_option, underlying, model, market_env)
    call_gamma = technique.gamma(call_option, underlying, model, market_env)
    call_vega = technique.vega(call_option, underlying, model, market_env)
    call_theta = technique.theta(call_option, underlying, model, market_env)
    call_rho = technique.rho(call_option, underlying, model, market_env)

    print("Call Option Greeks:")
    print(f"  Delta:   {call_delta:.4f}")
    print(f"  Gamma:   {call_gamma:.4f}")
    print(f"  Vega:    {call_vega:.4f}")
    print(f"  Theta:   {call_theta:.4f}")
    print(f"  Rho:     {call_rho:.4f}\n")

    # Compute Greeks for the put option
    put_delta = technique.delta(put_option, underlying, model, market_env)
    put_gamma = technique.gamma(put_option, underlying, model, market_env)
    put_vega = technique.vega(put_option, underlying, model, market_env)
    put_theta = technique.theta(put_option, underlying, model, market_env)
    put_rho = technique.rho(put_option, underlying, model, market_env)

    print("Put Option Greeks:")
    print(f"  Delta:   {put_delta:.4f}")
    print(f"  Gamma:   {put_gamma:.4f}")
    print(f"  Vega:    {put_vega:.4f}")
    print(f"  Theta:   {put_theta:.4f}")
    print(f"  Rho:     {put_rho:.4f}\n")

    # Compute implied volatilities using the computed option prices as targets
    call_iv = technique.implied_volatility(
        call_option, underlying, model, market_env, target_price=call_price
    )
    put_iv = technique.implied_volatility(
        put_option, underlying, model, market_env, target_price=put_price
    )

    print("Implied Volatilities:")
    print(f"  Call Option IV:  {call_iv:.4f}")
    print(f"  Put Option IV:   {put_iv:.4f}")

    # Plot a graph of price vs strike
    technique.graph(call_option, underlying, model, market_env)


if __name__ == "__main__":
    example_int()
