import numpy as np

def delta(S, K, r, T, vol, q, exerciseTimes=[],
               startDate=(), eps=10**-4, call=True, N=5000):
    """Return the delta of a option.

    Parameters
    ----------
    See document header.

    Returns
    -------
    delta_ : float
        The delta.
    
    Example(s)
    ----------
    >>> delta(95, 99, .08, 1, .2, .005, 'european call')
    >>> 0.6027636563788857

    >>> delta(100, 80, .08, 1, .2, .005, 'european put')
    >>> -0.0556145244008193
    
    """
    dT = T / N
    prUp = np.exp(vol * np.sqrt(dT))
    
    tree = BOPM(S, K, r, T, vol, q, exerciseTimes=[],
               startDate=(), eps=10**-4, call=True, N=5000, levels=1)

    opD, opU = tree[1]
    delta_ = (opU*prUp - opD*prUp) / (S * prUp**2 - S)
    
    return delta_

def gamma(S, K, r, T, vol, q, exerciseTimes=[],
               startDate=(), eps=10**-4, call=True, N=5000):
    """Return the gamma of an option.

    Parameters
    ----------
    See document header.

    Returns
    -------
    gamma_ : float
        The gamma.
    
    Example(s)
    ----------
    >>> gamma(95, 99, .08, 1, .2, .005, 'american call')
    >>> 0.020171905018027125

    """
    dT = T / N
    prUpSq = np.exp(2 * vol * np.sqrt(dT))
    
    tree = BOPM(S, K, r, T, vol, q, exerciseTimes=[],
               startDate=(), eps=10**-4, call=True, N=5000, levels=2)

    opDD, opDU, opUU = tree[2]
    opUD = opDU
    spUU, spDD = S*prUpSq, S/prUpSq
    
    A = opUU-opUD
    B = opUD-opDD
    X = spUU - S
    Y = S - spDD
    Z = spUU - spDD

    gamma_ = 2 * (A*Y - B*X) / (X*Y*Z)
    
    return gamma_

def theta(S, K, r, T, vol, q, exerciseTimes=[],
          startDate=(), eps=10**-4, call=True, N=5000, levels=2,
          normalize=True):
    """Return the theta of an option.

    For exact theta, initialize startDate, calander.py will then
    calculate the total number of trading days from startDate to T years
    in the future, and from that divide the total number of days by T,
    to accurately equidistrube theta each day in the options interval.

    Parameters
    ----------
    [See document header]
    normalize : bool, optional
        If theta will be normalized by days in the year.
        
    Returns
    -------
    theta_ : float
        The theta.
    
    Example(s)
    ----------
    >>> theta(95, 99, .08, 1, .2, .005, 'american call')
    >>> -0.028607269234589366

    >>> theta(95, 110, .08, 1, .2, .005, 'european put')
    >>> 0.006207352950055122

    """
    tree = BOPM(S, K, r, T, vol, q, exerciseTimes=[],
               startDate=(), eps=10**-4, call=True, N=5000, levels=2)

    opUD, op = tree[2][1], tree[0][0]
    dT = T / N
    theta_ = (opUD - op) / (2*dT)
    if normalize:
        days = calander.trDays(startDate, T) if startDate else 252
        T_ = T if startDate else 1
        theta_ = theta_ * T_ / days  #normalize to calculate theta per day
    
    return theta_

def vega(S, K, r, T, vol, q,
         optionType, exerciseTimes=[], N=500, eps=10**-4,
         startDate=(), deltaVol=10**-4):
    """Return the vega of an option.

    Parameters
    ----------
    See document header.
    deltaVol : float, optional
        The change in vol used to approximate vega.

    Returns
    -------
    vega_ : float
        The vega.
    
    Example(s)
    ----------
    >>> vega(100, 90, .08, .5, .2, 0, 'european call')
    >>> 15.260267051928622
    
    """
    diff = deltaVol / 2
    volA, volB = vol - diff, vol + diff
    
    prA = BOPM(S, K, r, T, volA, q,
                optionType, exerciseTimes, N, eps, startDate)
    prB = BOPM(S, K, r, T, volB, q,
                optionType, exerciseTimes, N, eps, startDate)
    
    vega_ = (prB - prA) / deltaVol

    return vega_

def freqVega(volA, prA, h, S, K, r, T, vol, q,
         optionType, exerciseTimes=[], N=500, eps=10**-4,
         startDate=(), deltaVol=10**-4):
    """Return the vega of an option.

    Parameters
    ----------
    See document header.
    deltaVol : float, optional
        The change in vol used to approximate vega.

    Returns
    -------
    vega_ : float
        The vega.
    
    Example(s)
    ----------
    >>> vega(100, 90, .08, .5, .2, 0, 'european call')
    >>> 15.260267051928622
    
    """
    diff = deltaVol / 2
    volB = volA + diff
    
    prB = BOPM(S, K, r, T, volB, q, optionType, exerciseTimes, N, eps, startDate)
    
    vega_ = (prB - prA) / deltaVol

    return vega_

def rho(S, K, r, T, vol, q,
         optionType, exerciseTimes=[], N=500, eps=10**-4,
         startDate=(), deltaR=10**-4):
    """Return the rho of an option.

    Parameters
    ----------
    See document header.
    deltaR : float, optional
        The change in intRate used to approximate vega.
        
    Returns
    -------
    rho_ : float
        The rho.
    
    Example(s)
    ----------
    >>> rho(100, 90, .08, .5, .3, 0, 'american call')
    >>> 31.070465134561687
    
    >>> rho(100, 110, .08, .5, .3, 0, 'bermuda put', [.15, .3, .45])
    >>> -22.15481826988608
    
    """
    diff = deltaR / 2
    rA, rB = intRate - diff, intRate + diff
    
    prA = BOPM(S, K, rA, T, vol, q,
                optionType, exerciseTimes, N, eps, startDate)
    prB = BOPM(S, K, rB, T, vol, q,
                optionType, exerciseTimes, height, eps, startDate)
    
    rho_ = (prB - prA) / deltaR
    
    return rho_

