def vol(S, K, r, T, q, optionType, opPr, prEr=10**(-4)):
    """No use of bsm"""
    call = isCall(optionType)
    seedVol = bsm.vol(spotPrice, strikePrice, intRate, T, dividend, opPr, call,
                   eps=prEr)
    seedPr = BOPM(spotPrice, strikePrice, intRate, T, seedVol, dividend,
                  optionType)

    error = abs(opPr - seedPr)
    if error > prEr:
        dVol = seedVol * prEr
        vega_ = vega(spotPrice, strikePrice, intRate, T, seedVol, dividend,
                     optionType)
        
    totEffort = 0
    while error > prEr and totEffort < 100:
        flag = -1 if vega_ < 0 else 1
        if opPr < seedPr:
            while opPr < seedPr:
                totEffort += 1
                seedVol -= flag * dVol
                seedPr = BOPM(spotPrice, strikePrice, intRate, T,
                              seedVol, dividend, optionType)
                error = abs(opPr - seedPr)

        else:
            while opPr > seedPr:
                totEffort += 1
                seedVol += flag*dVol
                seedPr = BOPM(spotPrice, strikePrice, intRate, T,
                              seedVol, dividend, optionType)
                error = abs(opPr - seedPr)
                
        dVol /= 10

    return seedVol

def volBSMSeed(S, K, r, T, q, optionType, opPr, prEr=10**(-4), maxEffort=100):
    """Use BSM implied volatiltiy to find the BOPM implied vol.

    Parameters
    ----------
    See document header.
    opPr : float
        Price of option.
    prEr : float, optional
        Accepted error in optionPrice error. Not the same as error in IV.
        
    Returns
    -------
    seedVol : float
        The BOPM/BSM implied volatiltiy.
    
    Example(s)
    ----------
    >>> vol()
    >>>
    
    """
    seedVol = bsm.vol(S, K, r, T, q, opPr, call, eps=prEr)
    seedPr = BOPM(S, K, r, T, seedVol, q, optionType)

    error = abs(opPr - seedPr)
    if error > prEr:
        dVol = seedVol * prEr
        vega_ = vega(spotPrice, strikePrice, intRate, T, seedVol, dividend,
                     optionType)
        
    totEffort = 0
    while error > prEr and totEffort < maxEffort:
        flag = -1 if vega_ < 0 else 1
        if opPr < seedPr:
            while opPr < seedPr:
                totEffort += 1
                seedVol -= flag * dVol
                seedPr = BOPM(spotPrice, strikePrice, intRate, T,
                              seedVol, dividend, optionType)
                error = abs(opPr - seedPr)

        else:
            while opPr > seedPr:
                totEffort += 1
                seedVol += flag*dVol
                seedPr = BOPM(spotPrice, strikePrice, intRate, T,
                              seedVol, dividend, optionType)
                error = abs(opPr - seedPr)
                
        dVol /= 10

    return seedVol
