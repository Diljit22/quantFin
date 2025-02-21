
from src.instruments.european_option import EuropeanOption
from src.market.market_environment import MarketEnvironment
from src.techniques.closed_forms.blacks_aprx import FD_Black
from src.underlyings.stock import Stock
import numpy as np

tech = FD_Black()
intr = EuropeanOption(strike=55, maturity=.5, is_call=True)
my_stock = Stock(spot=50, volatility=.3, dividend=0)
my_stock.discrete_dividend = np.array([.7, .7])
my_stock.dividend_times = np.array([[3/12, 5/12]])
my_r = MarketEnvironment(rate=.1)

x = tech.price(intr, my_stock, 8, my_r)
print(x)
iv_ = tech.implied_volatility(intr, my_stock, 8, my_r, x)
print(iv_)
print("=============")

intr = EuropeanOption(strike=50, maturity=.5, is_call=False)
my_stock = Stock(spot=55, volatility=.3, dividend=0)
my_stock.discrete_dividend = np.array([.7, .7])
my_stock.dividend_times = np.array([[3/12, 5/12]])
my_r = MarketEnvironment(rate=.1)

x = tech.price(intr, my_stock, 8, my_r)
print(x)
print("=============")
tech = FD_Black()
intr = EuropeanOption(strike=55, maturity=.5, is_call=True)
my_stock = Stock(spot=50, volatility=.1, dividend=0)
my_stock.discrete_dividend = np.array([.1, 10])
my_stock.dividend_times = np.array([[3/12, 5/12]])
my_r = MarketEnvironment(rate=.1)

x = tech.price(intr, my_stock, 8, my_r)
print(x)
iv_ = tech.implied_volatility(intr, my_stock, 8, my_r, x)
print(iv_)