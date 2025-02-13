
###Greeks minus gamma
def delta(S, K, r, T, vol, q,
         optionType, exerciseTimes=[], height=400,
         eps=10**-4, startDate=(), deltaSpot=10**-5):
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
    >>> 0.6142208556880746

    >>> delta(100, 80, .08, 1, .2, .005, 'european put')
    >>> -0.05778247273990899
    
    """
    diff = deltaSpot / 2
    spA, spB = spotPrice - diff, spotPrice + diff
    
    prA = TOPM(spA, strikePrice, intRate, T, vol, dividend,
                optionType, exerciseTimes, height, eps, startDate)
    prB = TOPM(spB, strikePrice, intRate, T, vol, dividend,
                optionType, exerciseTimes, height, eps, startDate)
    
    delta_ = (prB - prA) / deltaSpot
  
    return delta_

def theta(spotPrice, strikePrice, intRate, T, vol, dividend,
         optionType, exerciseTimes=[], height=400,
         eps=10**-4, startDate=(), deltaT=10**-3, normalize=True):
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
    >>> -0.028619027867760577

    >>> theta(95, 110, .08, 1, .2, .005, 'european put')
    >>> 0.006211160561653978

    """
    op = TOPM(spotPrice, strikePrice, intRate, T, vol, dividend,
                optionType, exerciseTimes, height, eps, startDate)
    theta_ = (op[1]-op[0]) / (2*T/height)

    if normalize:
        days = calander.trDays(startDate, T) if startDate else 252
        T_ = T if startDate else 1
        theta_ = theta_ * T_ / days  #normalize to calculate theta per day

    return theta_

def vega(spotPrice, strikePrice, intRate, T, vol, dividend,
         optionType, exerciseTimes=[], height=500, eps=10**-4,
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
    >>> 15.634841935607824
    
    """
    diff = deltaVol / 2
    volA, volB = vol - diff, vol + diff
    
    prA = TOPM(spotPrice, strikePrice, intRate, T, volA, dividend,
                optionType, exerciseTimes, height, eps, startDate)
    prB = TOPM(spotPrice, strikePrice, intRate, T, volB, dividend,
                optionType, exerciseTimes, height, eps, startDate)
    
    vega_ = (prB - prA) / deltaVol

    return vega_

def rho(spotPrice, strikePrice, intRate, T, vol, dividend,
         optionType, exerciseTimes=[], height=500, eps=10**-4,
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
    >>> 31.076106565244288
    
    >>> rho(100, 110, .08, .5, .3, 0, 'bermuda put', [.15, .3, .45])
    >>> -22.497356868189655
    
    """
    diff = deltaR / 2
    rA, rB = intRate - diff, intRate + diff
    
    prA = TOPM(spotPrice, strikePrice, rA, T, vol, dividend,
                optionType, exerciseTimes, height, eps, startDate)
    prB = TOPM(spotPrice, strikePrice, rB, T, vol, dividend,
                optionType, exerciseTimes, height, eps, startDate)
    
    rho_ = (prB - prA) / deltaR
    
    return rho_
