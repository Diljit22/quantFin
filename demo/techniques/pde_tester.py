# --- Example Usage ---
from re import M
from src.instruments.european_option import EuropeanOption
from src.market.market_environment import MarketEnvironment
from src.models.black_scholes_merton import BlackScholesMerton
from src.underlyings.stock import Stock
from src.techniques.pde.pde_techique import PDETechnique

if __name__ == "__main__":

    # Create example objects.
    underlying = Stock(spot=100.0, volatility=0.2, dividend=0.02)
    market_env = MarketEnvironment(rate=0.05)
    model = BlackScholesMerton(0.2)
    pde_solver = PDETechnique(S_max=600.0, M=256, N=256)

    # Example option parameters: a list of (strike, maturity) pairs.
    option_parameters = [(90, 1.0), (100, 0.75), (110, 0.5)]
    for strike, maturity in option_parameters:
        call_option = EuropeanOption(strike=strike, maturity=maturity, is_call=True)
        put_option = EuropeanOption(strike=strike, maturity=maturity, is_call=False)
        call_price = pde_solver.price(call_option, underlying, model, market_env)
        put_price = pde_solver.price(put_option, underlying, model, market_env)

        print(f"Option parameters: Strike = {strike:.1f}, Maturity = {maturity:.2f}")
        print(f"  -> Call Price = {call_price:.4f}")
        print(f"  -> Put Price  = {put_price:.4f}")
        print("-------------------------------------------------------")
