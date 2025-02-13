"""Implement recombining trinomial pricing model"""

import numpy as np
from priceEU import priceEU #need to change name of priceEU

def priceBE(S, K, r, T, priceUp, probJumps, exTimes, unitSpan,
            depth=5000, call=True, levels=0):
    """Price a Bermuda option via the a recombining trinom tree.

    PriceJumps = X, 1, 1/X
    priceUp = X

    Parameters
    ----------
    S       : float
    K       : float
    r       : float
    T       : float
    exTimes : Time in year until START of day where op can be excer.
    unitSpan: Time in years one day is. (See canClaim for this 1/252 works)
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

    """START BERM-SPECFIC"""
    #See if can excersise which choice of depth.
    canClaim = ableToClaim(dT, depth, exTimes, unitSpan)
    if not canClaim:
        return priceEU(S, K, r, T, priceUp, probUp, depth, call, levels)
    """END BERM-SPECFIC"""
    
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

        """START BERM-SPECFIC"""
        if i in not in canClaim:
            opPr = np.maximum(opPr[:M], 0)

        else:
            J = depth-i
            ex = np.maximum(S[J:-J]-K, 0) if call else np.maximum(K-S[J:-J], 0)
            opPr = np.maximum(opPr[:-1], ex)
        """END BERM-SPECFIC"""

        if levels and i < levels:
            rowsOut[i] = opPr[:M]
   
    return opPr[0] if levels == 0 else rowsOut
