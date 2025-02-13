import math
from typing import Union


def lower_bound_rate(
    call_price: float, put_price: float, S: float, K: float, T: float
) -> float:
    """
    Compute a lower bound on the risk-free rate from put-call inequality:
       C - P <= S - K e^{-rT}

    => r >= [ln(K) - ln(S - C + P)] / T

    Parameters
    ----------
    call_price : float
        Observed call price
    put_price : float
        Observed put price
    S : float
        Underlying spot
    K : float
        Strike price
    T : float
        Time to maturity (years)

    Returns
    -------
    float
        Lower bound on r from the inequality.

    Raises
    ------
    ValueError
        If S - call_price + put_price <= 0 => log not defined.

    Example
    -------
    >>> r_min = lower_bound_rate(0.5287, 6.7143, 100, 110, 0.5)
    >>> print(r_min)  # ~ 0.07
    """
    val = S - call_price + put_price
    if val <= 0.0:
        raise ValueError("S - call_price + put_price <= 0 => cannot compute log.")
    return -math.log(val / K) / T


def implied_rate(
    call_price: float,
    put_price: float,
    S: float,
    K: float,
    T: float,
    q: Union[float, None] = 0.0,
    eps: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """
    Solve numerically for the implied risk-free interest rate `r` from put-call
    parity. For continuous dividend yield q:
       C - P = S e^{-qT} - K e^{-rT}.

    If discrete dividends, you must adapt the approach or
    pass an adjusted S.

    Parameters
    ----------
    call_price : float
        Observed call price
    put_price : float
        Observed put price
    S : float
        Underlying spot
    K : float
        Strike
    T : float
        Maturity in years
    q : float or None, optional, default=0.0
        Continuous dividend yield
    eps : float, default=1e-6
        Convergence tolerance
    max_iter : int, default=100
        Max iterations for bisection or secant approach

    Returns
    -------
    float
        The implied interest rate r.

    Raises
    ------
    ValueError
        If we cannot bracket a solution or if discrete dividends are not handled.

    Example
    -------
    >>> r_est = implied_rate(0.5287, 6.7143, 100, 110, 0.5, q=0.01)
    >>> print(r_est)  # ~ 0.08
    """
    if q is None:
        q = 0.0

    left_side = call_price - put_price
    discounted_spot = S * math.exp(-q * T)

    def f(r_val: float) -> float:
        disc_k = K * math.exp(-r_val * T)
        return (discounted_spot - disc_k) - left_side

    # We bracket r in [r_low, r_high]
    r_low, r_high = -1.0, 1.0
    f_low, f_high = f(r_low), f(r_high)
    # Expand bracket if necessary
    while f_low * f_high > 0:
        if abs(f_low) < abs(f_high):
            r_low -= 0.5
            f_low = f(r_low)
            if r_low < -100:
                raise ValueError(
                    "Cannot bracket negative rate further. Possibly no solution."
                )
        else:
            r_high += 0.5
            f_high = f(r_high)
            if r_high > 2.0:
                raise ValueError(
                    "Cannot bracket positive rate further. Possibly no solution."
                )

    # Bisection
    for _ in range(max_iter):
        mid = 0.5 * (r_low + r_high)
        fm = f(mid)
        if abs(fm) < eps:
            return mid
        f_low = f(r_low)
        if f_low * fm < 0:
            r_high = mid
        else:
            r_low = mid
    return 0.5 * (r_low + r_high)
