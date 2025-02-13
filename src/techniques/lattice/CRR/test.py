""" test """

from price import price
import numpy as np
from priceAM import priceAM

def priceJumps(vol, dT):
    up = np.exp(vol * np.sqrt(dT))
    return up

def probJumps(r, q, dT, priceUp):
    priceDown = 1/priceUp
    up = (np.exp((r-q)*dT) - priceDown) / (priceUp - priceDown)
    return up


# Example parameters
S = 100  # Spot price
K = 100  # Strike price
r = 0.05  # Risk-free rate
T = 1  # Time to maturity in years
vol = 0.2  # Volatility
q = 0.02  # Dividend yield
N = 50  # Number of steps
call = True  # Call option
depth = N
dT = T/depth
priUp = priceJumps(vol, dT)
proUp = probJumps(r, q, dT, priUp)

A = priceAM(S, K, r, T, priUp, proUp, depth, call=True)
print(A)
9.040821700571652

A = priceAM(S, K, r, T, priUp, proUp, depth, call=False)
print(A)
5.11976561573942

