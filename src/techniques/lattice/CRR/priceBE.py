"""Implement recombining Binom Lattice to price a Bermuda Option."""

import numpy as np
from priceEU import priceEU #need to change name of priceEU

def priceBE(S, K, r, T, priceUp, probUp, exTimes, unitSpan    
            depth=5000, call=True, levels=0):
    """Price a Bermuda option via the a recombining binomial tree.

    Parameters
    ----------
    S       : float
    K       : float
    r       : float
    T       : float
    exTimes : Time in year until START of day where op can be excer.
    unitSpan: Time in years one day is. (See canClaim for this 1/252 works)
    priceUp : float : func fixed with req. params so float
    probUp  : float : func fixed with req. params so float
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
    
    #Value at expiry
    W = np.zeros((2, depth+1))
    W[0] = np.arange(-depth, depth+1, 2, dtype=float)
    W[1] = np.arange(-depth+1, depth+2, 2, dtype=float)
    S *= priceUp ** W
    
    opPr = np.maximum(S[0] - K, 0) if call else np.maximum(K - S[0], 0)
    rowsOut = [1] * levels
    for i in np.arange(depth-1, -1, -1):
        M = i+1
        opPr[:M] = disc * (probUp * np.ediff1d(opPr[:i+2]) + opPr[:i+1])

        """START BERM-SPECFIC"""
        if i in not in canClaim:
            opPr = np.maximum(opPr[:-1], 0)

        else:
            row = (depth+i) % 2
            A, B = (depth-i)//2, (depth-i+1)//2
            if call:
                ex = np.maximum(S[row][A:-B] - K, 0)
            else:
                ex = np.maximum(K - S[row][A:-B], 0)
            
            opPr = np.maximum(opPr[:-1], ex)
        """END BERM-SPECFIC"""
            
        if levels and i < levels:
            rowsOut[i] = opPr[:M]
            
    return opPr[0] if levels == 0 else rowsOut


