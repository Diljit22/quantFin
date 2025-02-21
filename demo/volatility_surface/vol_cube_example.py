"""
vol_cube_example.py
===================

Demonstrates how to build an implied volatility surface
using the VolCube class. The volatility surface in this example is
constructed by comparing the market prices for options with model prices
computed using the Black-Scholes-Merton closed-form technique. The class
is much more general!

The VolCube class builds a surface (returned as a pandas DataFrame)
that contains implied volatilities.
"""

from src.volatility_surface.vol_cube import VolCube
from src.techniques.closed_forms.bsm_technique import BlackScholesMertonTechnique
from src.instruments.european_option import EuropeanOption
from src.underlyings.stock import Stock
from src.market.market_environment import MarketEnvironment
from src.models.black_scholes_merton import BlackScholesMerton

# Define the set of strikes and maturities for the surface.
strikes = [80, 90, 100, 110, 120]
maturities = [0.5, 1.0, 2.0]

# Example market prices for each (strike, maturity) combination.
market_prices = {
    (80, 0.5): 21.50,
    (90, 0.5): 14.10,
    (100, 0.5): 8.20,
    (110, 0.5): 4.50,
    (120, 0.5): 2.30,
    (80, 1.0): 24.50,
    (90, 1.0): 17.10,
    (100, 1.0): 11.20,
    (110, 1.0): 7.00,
    (120, 1.0): 4.20,
    (80, 2.0): 30.50,
    (90, 2.0): 22.10,
    (100, 2.0): 16.20,
    (110, 2.0): 12.00,
    (120, 2.0): 8.20,
}

# Create necessary objects
technique = BlackScholesMertonTechnique(cache_results=True)
underlying = Stock(spot=100.0, volatility=0.20, dividend=0.01)
model = BlackScholesMerton(sigma=0.20)
market_env = MarketEnvironment(rate=0.05)

# Create the VolCube instance.
# The 'is_call' flag indicates whether the options are calls (True) or puts (False).
vcube = VolCube(technique, underlying, model, market_env, is_call=True)

# Build the volatility surface.
# - strikes and maturities specify the grid.
# - market_prices is a dictionary with keys as (strike, maturity) pairs.
# - parallel=True enables parallel processing with max_workers set to 6.
# - tol and max_iter are parameters for the implied volatility solver.
df_surface = vcube.build(
    strikes=strikes,
    maturities=maturities,
    market_prices=market_prices,
    parallel=True,
    max_workers=6,
    tol=1e-8,
    max_iter=200,
)

# Print the volatility surface
print("Implied Volatility Surface:")
print(df_surface)
