"""Implement recombining trinomial pricing model"""

import numpy as np

def priceAM(S, K, r, T, priceUp, probJumps, depth=5000, call=True, levels=0):
    """Price a American option via the a recombining trinom tree.

    PriceJumps = X, 1, 1/X
    priceUp = X

    Parameters
    ----------
    S       : float
    K       : float
    r       : float
    T       : float
    priceUp : float  : func fixed with req. params so float
    probJumps  : arr  : func fixed with req. params so float
    depth   : int
    call    : bool 
    levels  : int

    Returns
    -------
    

    Example(s)
    ---------
    >>> 

    """
    dT = T / depth
    disc = np.exp(-r * dT)
    pU, pS, pD = probJumps

    #Value at expiry
    S *= priceUp ** np.arange(-depth, depth+1, dtype=float)
    opPr = np.maximum(S - K, 0) if call else np.maximum(K - S, 0)

    rowsOut = [1] * levels
    #Value at earlier times
    for i in np.arange(depth-1, -1, -1):
        M = 2*i+1
        opPr[:M] = disc * (pU*opPr[2:M+2] + pS*opPr[1:M+1] + pD*opPr[:M])

        J = depth-i
        ex = np.maximum(S[J:-J] - K, 0) if call else np.maximum(K - S[J:-J], 0)
        opPr = np.maximum(opPr[:M], ex)

        if levels and i < levels:
            rowsOut[i] = opPr[:M]
   
    return opPr[0] if levels == 0 else rowsOut


        # Backward induction with early exercise
        for i in range(N - 1, -1, -1):
            assetPrices = assetPrices[:-1] / u
            intrinsicValue = np.maximum(0, assetPrices - K) if call else np.maximum(0, K - assetPrices)
            payoff = np.maximum(intrinsicValue, discount * (pu * payoff[:-2] + pm * payoff[1:-1] + pd * payoff[2:]))
        
