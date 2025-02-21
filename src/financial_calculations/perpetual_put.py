"""
perpetual_put.py
================

Provides a function to price a perpetual put option.
The pricing is derived by solving an associated ODE.

Functions
---------
perpetual_put(S, K, r, vol, q)
    Price a put that never expires.

"""


def perpetual_put(S, K, r, vol, q):
    """
    Price an American put that never expires.

    Computes the price of a perpetual put option (an American put option without
    expiration) using a closed-form solution derived from the associated ODE.

    Parameters
    ----------
    S : float
        Current price of the stock.
    K : float
        Strike price of the option.
    r : float
        Annualized risk-free interest rate, continuously compounded.
    vol : float
        Volatility of the stock.
    q : float
        Continuous dividend yield.

    Returns
    -------
    value : float
        Value of the perpetual put option.

    Notes
    -----
    The model assumes r != 0 because the derivation involves solving a quadratic
    equation of the form (x^2 + bx + c) = 0. A zero risk-free rate would lead to a
    trivial constant term and an undefined solution.

    Examples
    --------
    >>> perpetual_put(S=150, K=100, r=0.08, vol=0.2, q=0.005)
    1.8344292693352158
    """
    volSq = vol**2

    # Solve the associated quadratic equation.
    const = r - q - volSq / 2
    discr = const**2 + 2 * r * volSq
    root = -(const + discr**0.5) / volSq

    # Compute the solution to the ODE.
    mnRoot = root - 1
    mul = -K / mnRoot
    base = (mnRoot / root) * (S / K)
    value = mul * (base**root)

    return value
