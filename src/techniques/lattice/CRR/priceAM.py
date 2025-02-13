"""Implement recombining Binom Lattice to price a American Option."""

import numpy as np

def priceAM(S, K, r, T, priceUp, probUp, depth=5000, call=True, levels=0):
    """Price a American option via the a recombining binomial tree.

    Parameters
    ----------
    S       : float
    K       : float
    r       : float
    T       : float
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

        row = (depth+i) % 2
        A, B = (depth-i)//2, (depth-i+1)//2
        if call:
            ex = np.maximum(S[row][A:-B] - K, 0)
        else:
            ex = np.maximum(K - S[row][A:-B], 0)
            
        opPr = np.maximum(opPr[:-1], ex)
        if levels and i < levels:
            rowsOut[i] = opPr[:M]
            
    return opPr[0] if levels == 0 else rowsOut


