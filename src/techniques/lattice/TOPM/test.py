""" test """

from price import price
import numpy as np
from priceAm import priceAM

def priceJumps(vol, dT):
    u = np.exp(vol * np.sqrt(2*dT))
    return [u, 1, 1/u]

def probJumps(r, vol, q, dT):
    
    dT_ = dT/2
    drift = (r-q) * dT_
    noise = vol * np.sqrt(dT_)

    A = np.exp(drift + noise)
    B = np.exp(2*noise)
    norm = np.square(B - 1)
    
    pU = np.square(A - 1) / norm
    pD = np.square(B - A) / norm
    pS = 1 - pU - pD
    
    return [pU, pS, pD]

S, K, r, T = 50, 52, .05, 2
q, vol, depth = 0, .3, 5000

dT = T/depth
priUp = priceJumps(vol, dT)[0]
proJ = probJumps(r, vol, q, dT)

A = price(S, K, r, T, priUp, proJ, depth)
print(A)
9.708751978361748

A = price(S, K, r, T, priUp, proJ, depth, call=False)
print(A)
6.760297716195268

"""American"""
S, K, r, T = 50, 52, .05, 2
q, vol, depth = 0, .3, 5000

dT = T/depth
priUp = priceJumps(vol, dT)[0]
proJ = probJumps(r, vol, q, dT)

A = priceAM(S, K, r, T, priUp, proJ, depth)
print(A)
9.708751978361748

A = priceAM(S, K, r, T, priUp, proJ, depth, call=False)
print(A)
7.472077507429825
