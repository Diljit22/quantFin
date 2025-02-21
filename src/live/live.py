"""
live.py
=======
This file defines a live pricing function that uses data from FRED and Polygon to build
the pricing environment and then prices an option using a chosen model and technique.

0. made model is input
1. make pricing enviroment
2. price.

"""

import math
from io.fred_provider import FredDataProvider
from io.polygon_provider import PolygonDataProvider
from src.instruments.european_option import EuropeanOption
from src.underlyings.stock import Stock
from src.market.market_environment import MarketEnvironment


# A dummy model for demonstration purposes.
class DummyModel:
    def __init__(self, **kwargs):
        # Dummy BSM model; add parameters as needed.
        pass


def live_price(symbol: str, model_cls, tech_cls, **kwargs) -> float:
    """
    Price an option live using FRED and Polygon data.

    Parameters:
      symbol   : str
          The Polygon option symbol.
      model_cls: class
          The model class to instantiate (e.g., DummyModel).
      tech_cls : class
          The pricing technique class to instantiate.
      **kwargs : Additional keyword arguments such as:
                 - strike: Option strike.
                 - maturity: Option maturity (years).
                 - option_type: "Call" or "Put".
                 - fred_api_key: API key for FRED.
                 - polygon_api_key: API key for Polygon.

    Returns:
      float: The computed option price.
    """
    # 1. Extract risk–free rate from FRED.
    fred_provider = FredDataProvider(**kwargs)
    r = fred_provider.getRiskFreeRate()
    if r is None:
        raise ValueError("Risk–free rate data is missing from FRED.")

    # 2. Extract market data from Polygon.
    polygon_provider = PolygonDataProvider(**kwargs)
    market_data = polygon_provider.getMarketData(symbol)
    if "spot" not in market_data or "volatility" not in market_data:
        raise ValueError(
            f"Missing required market data for symbol {symbol} from Polygon."
        )
    S = market_data["spot"]
    sigma = market_data["volatility"]
    dividend = market_data.get("dividend", 0.0)
    discrete_dividend = market_data.get("discrete_dividend", None)
    dividend_times = market_data.get("dividend_times", None)

    # 3. Create instrument, underlying, and market environment.
    strike = kwargs.get("strike", 100)
    maturity = kwargs.get("maturity", 1)
    option_type = kwargs.get("option_type", "Call")
    instrument = EuropeanOption(strike, maturity, option_type == "Call")
    my_stock = Stock(S, sigma, dividend)
    if discrete_dividend is not None and dividend_times is not None:
        my_stock.discrete_dividend = discrete_dividend
        my_stock.dividend_times = dividend_times
    market_env = MarketEnvironment(r)

    # 4. Instantiate model.
    model = model_cls(**kwargs)

    # 5. Prepare keyword arguments for the pricing technique.
    # Remove keys that belong to the instrument, which FD_BSM doesn't expect.
    tech_kwargs = kwargs.copy()
    for key in ["strike", "maturity", "option_type"]:
        tech_kwargs.pop(key, None)

    if tech_cls is None:
        # Default to closed-form BSM technique.
        from src.techniques.closed_forms.bsm_finite_diff import FD_BSM

        tech = FD_BSM(**tech_kwargs)
    else:
        tech = tech_cls(**tech_kwargs)

    # 6. Price the option.
    price = tech.price(instrument, my_stock, model, market_env)
    return price


if __name__ == "__main__":
    # Example usage:
    # Assume the option symbol is "POLY_OPTION_SYMBOL" and we set strike, maturity, etc.
    live_option_price = live_price(
        "POLY_OPTION_SYMBOL",
        model_cls=DummyModel,
        tech_cls=None,
        strike=100,
        maturity=1,
        option_type="Call",
        fred_api_key="YOUR_FRED_KEY",
        polygon_api_key="YOUR_POLYGON_KEY",
    )
    print("Live Price =", live_option_price)
