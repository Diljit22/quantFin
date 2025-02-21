from src.instruments.european_option import EuropeanOption
from src.market.market_environment import MarketEnvironment
from src.techniques.closed_forms.bsj_closed_form import FD_BSMJ
from src.underlyings.stock import Stock

S = 100 # current stock price
T = 1 # time to maturity
r = 0.02 # risk free rate
m = 0 # meean of jump size
v = 0.3 # standard deviation of jump
lam = 1 # intensity of jump i.e. number of jumps per annum
steps =255 # time steps
Npaths =200000 # number of paths to simulate
sigma = 0.2 # annaul standard deviation , for weiner process
K =100


my_tech = FD_BSMJ()
my_stock = Stock(S, sigma, 0)
my_op = EuropeanOption(K, T, True)
my_r = MarketEnvironment(r)
import numpy as np

class dummyModel:
    def __init__(self):
        self.lam = lam
        # Set kappa = exp(m + 0.5*v^2) - 1 for consistency with Merton's jump diffusion.
        self.kappa = np.exp(m + 0.5 * v**2) - 1  
        self.delta_j = v

my_model = dummyModel()

# Now your FD_BSMJ pricing should be consistent:
x = my_tech.price(my_op, my_stock, my_model, my_r)
print(x)
iv_ = my_tech.implied_volatility(my_op, my_stock, my_model, my_r, x)
print(iv_)


import numpy as np
from scipy.stats import norm
import math
from scipy.optimize import minimize_scalar   
N = norm.cdf

def BS_CALL(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * N(d1) - K * np.exp(-r*T)* N(d2)

def BS_PUT(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma* np.sqrt(T)
    return K*np.exp(-r*T)*N(-d2) - S*N(-d1)    
    

def merton_jump_call(S, K, T, r, sigma, m , v, lam):
    p = 0
    for k in range(40):
        r_k = r - lam*(m-1) + (k*np.log(m) ) / T
        sigma_k = np.sqrt( sigma**2 + (k* v** 2) / T)
        k_fact = math.factorial(k)
        p += (np.exp(-m*lam*T) * (m*lam*T)**k / (k_fact))  * BS_CALL(S, K, T, r_k, sigma_k)
    
    return p 

def merton_jump_put(S, K, T, r, sigma, m , v, lam):
    p = 0 # price of option
    for k in range(40):
        r_k = r - lam*(m-1) + (k*np.log(m) ) / T
        sigma_k = np.sqrt( sigma**2 + (k* v** 2) / T)
        k_fact = np.math.factorial(k) # 
        p += (np.exp(-m*lam*T) * (m*lam*T)**k / (k_fact)) \
                    * BS_PUT(S, K, T, r_k, sigma_k)
    return p 

S = 100 # current stock price
T = 1 # time to maturity
r = 0.02 # risk free rate
m = 0 # meean of jump size
v = 0.3 # standard deviation of jump
lam = 1 # intensity of jump i.e. number of jumps per annum
sigma = 0.2 # annaul standard deviation , for weiner process
K =100

cf_price =  merton_jump_call(S, K, T, r, sigma, np.exp(m+v**2*0.5) , v, lam)

print('Merton Price =', cf_price)
print('Black Scholes Price =', BS_CALL(S,K,T,r, sigma))