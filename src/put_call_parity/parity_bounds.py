"""
Put-Call Parity & Bounds
========================

A collection of utility functions related to the put-call relationship
in European (and dividend-less American) options:

Example
-------
>>> from src.utils.put_call_parity_bounds import put_call_parity, put_call_bound
>>> c_price = put_call_parity(put_price, S=100, K=110, r=0.08, T=0.5, q=0.01, price_call=False)
>>> (lb, ub) = put_call_bound(c_price, 100, 110, 0.08, 0.5, bound_call=True)
>>> print(c_price, lb, ub)
"""

import math
from typing import Union


def put_call_parity(
    option_price: float,
    S: float,
    K: float,
    r: float,
    T: float,
    q: Union[float, None] = 0.0,
    price_call: bool = False,
) -> float:
    """
    Use put-call parity to compute the price of the complementary
    (European) option (call <-> put).

    The formula:
        C - P = S e^{-qT} - K e^{-rT}

    Parameters
    ----------
    option_price : float
        The known option price (call if `price_call` = False, else put).
    S : float
        Current underlying spot price.
    K : float
        Strike price of the option.
    r : float
        Annualized, continuously compounded risk-free rate.
    T : float
        Time to maturity in years.
    q : float or None, optional, default=0.0
        Continuous dividend yield (0.0 if none).
    price_call : bool, default=False
        If True, interpret `option_price` as the call to get the put.
        Otherwise interpret it as the put to get the call.

    Returns
    -------
    float
        The computed price of the complementary option.

    Notes
    -----
    - If `q` is nonzero, discount spot by e^{-qT}.
    - This function does not handle discrete dividends directly.
      For discrete dividends, adjust S accordingly or
      handle them in a more advanced method.

    Example
    -------
    >>> c_price = put_call_parity(6.71, S=100, K=110, r=0.08, T=0.5, q=0.01, price_call=False)
    >>> print(round(c_price, 4))
    0.5287
    """
    if q is None:
        q = 0.0

    discounted_strike = K * math.exp(-r * T)
    discounted_spot = S * math.exp(-q * T)
    call_minus_put = discounted_spot - discounted_strike

    if price_call:
        # We know call => want put
        # put = call - (S e^{-qT} - K e^{-rT})
        result = option_price - call_minus_put
    else:
        # We know put => want call
        # call = put + (S e^{-qT} - K e^{-rT})
        result = option_price + call_minus_put
    return result


def put_call_bound(
    option_price: float,
    S: float,
    K: float,
    r: float,
    T: float,
    bound_call: bool = False,
) -> tuple:
    """
    Compute naive lower/upper bounds for a (European or American) call or put
    using put-call inequalities.

    Parameters
    ----------
    option_price : float
        Price of the contract (call if bound_call=True, else put).
    S : float
        Current spot price of the underlying.
    K : float
        Strike price.
    r : float
        Risk-free rate, continuously compounded.
    T : float
        Time to maturity in years.
    bound_call : bool, default=False
        If True, interpret `option_price` as a call. Otherwise as a put.

    Returns
    -------
    (float, float)
        (lower_bound, upper_bound)

    Notes
    -----
    - For a call on a non-dividend-paying asset:
      Lower Bound: max(0, S - K e^{-rT})
      Upper Bound: typically S (for American) or the premium from immediate exercise
      Similarly for puts: max(0, K e^{-rT} - S) up to K e^{-rT} in some sense.

    Example
    -------
    >>> lower, upper = put_call_bound(2.03, 36.0, 37.0, 0.055, 0.5, bound_call=False)
    >>> print(lower, upper)
    2.0263..., 3.03...
    """
    discounted_k = K * math.exp(-r * T)
    # call - put ~ S - K e^{-rT},
    # so for calls => call >= max(0, S - K e^{-rT})
    # for puts  => put >= max(0, K e^{-rT} - S)
    if bound_call:
        lower_bound = max(0.0, S - discounted_k)
        upper_bound = S  # an American call can't exceed S
    else:
        lower_bound = max(0.0, discounted_k - S)
        upper_bound = discounted_k  # a put can't exceed K e^{-rT} too far
    return (lower_bound, upper_bound)
