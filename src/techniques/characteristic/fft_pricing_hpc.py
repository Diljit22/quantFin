#!/usr/bin/env python3
# cm_fft_final.py

"""
Carr-Madan FFT (HPC) with Domain Shifting for European Option Pricing
=====================================================================
A final "top-tier" demonstration that ensures non-zero prices for both
in-the-money and out-of-the-money strikes by:

1. Normalizing the random variable: X = ln(S_T / S_0).
2. Using log-strike domain k in [-B/2, +B/2], so that k=0 -> K=S_0.
3. Possibly a partial Simpson weighting to avoid near-zero issues.
4. (Optional) Concurrency in building the transform if 'use_parallel=True'.

We show a standard BSM characteristic function for demonstration, but
you can plug in any advanced CF. This code is typically how a robust
FFT approach is implemented to get correct prices for K < S_0 or K > S_0.
"""

import math
import cmath
import numpy as np
import concurrent.futures
from typing import Callable, List, Tuple


def bsm_char_func_normalized(
    r: float, q: float, sigma: float, T: float
) -> Callable[[complex], complex]:
    """
    BSM characteristic function for X=ln(S_T/S_0). i.e. ignoring ln(S0).

    X ~ (r-q-0.5*sigma^2)*T + sigma* sqrt(T)*Z
    => phi_X(u) = E[ exp(i u X) ] = exp(i u drift - 0.5 var * u^2 )
      with drift=(r-q-0.5*sigma^2)*T, var=sigma^2*T
    """
    drift = (r - q - 0.5 * sigma * sigma) * T
    var = sigma * sigma * T

    def phi(u: complex) -> complex:
        # exponent: i*u*(drift) + -0.5*var*(u^2)
        return cmath.exp(1j * u * drift - 0.5 * var * (u * u))

    return phi


def cm_fft_shifted(
    phi_func: Callable[[complex], complex],
    r: float,
    T: float,
    alpha: float,
    N: int,
    B: float,
    use_parallel: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    HPC Carr-Madan single-lap approach over k in [-B/2 .. B/2].
    k_j = k_min + j*delta, where k_min=-B/2, k_max=+B/2, delta=B/N.
    Then K_j = S0 * exp(k_j).

    G_j = e^{-rT} * e^{-i u_j k_min} * phi(u_j - i(alpha+1)) / denominator(u_j),
    denominator(u)= alpha(alpha+1) - u^2 + i(2alpha+1)*u

    We apply a partial Simpson weighting for G_j prior to FFT, which helps
    with in-the-money option values.

    Returns
    -------
    (call_vals, k_array):
      call_vals[j] => price of call at log-strike k_j
      k_array[j]   => the domain in [-B/2.. +B/2]
    """
    delta_k = B / N
    k_min = -0.5 * B
    k_vals = k_min + np.arange(N) * delta_k
    lam = math.pi / B
    j_idx = np.arange(N)
    u_vals = j_idx * lam

    discount = math.exp(-r * T)

    def denom(u: float) -> complex:
        return alpha * (alpha + 1.0) - (u * u) + 1j * (2.0 * alpha + 1.0) * u

    G = np.zeros(N, dtype=complex)

    def compute_chunk(start: int, end: int):
        for j in range(start, end):
            u = u_vals[j]
            # shift => (u - i(alpha+1))
            u_shift = complex(u, -(alpha + 1.0))
            phi_val = phi_func(u_shift)
            dnm = denom(u)
            if abs(dnm) < 1e-14:
                G[j] = 0.0
            else:
                shift_factor = cmath.exp(-1j * u * k_min)  # e^{- i u (k_min)}
                G[j] = discount * shift_factor * (phi_val / dnm)

    if use_parallel and N > 256:
        chunk_size = N // 4
        tasks = []
        with concurrent.futures.ThreadPoolExecutor() as ex:
            for c in range(4):
                st = c * chunk_size
                en = N if c == 3 else (c + 1) * chunk_size
                tasks.append(ex.submit(compute_chunk, st, en))
            concurrent.futures.wait(tasks)
    else:
        compute_chunk(0, N)

    # partial Simpson weighting => for j even => 2, for j odd => 4
    # or do a simpler approach. We do a "Simpson-ish" approach for better
    # results near in-the-money. We'll do:
    #   w_j = 3 if j%2 !=0, else 2, but half for endpoints, etc.
    # For a simpler approach, just do trapezoid. We'll do a minor variant:
    w = np.ones(N, dtype=float) * 2.0
    w[0] = 1.0
    w[-1] = 1.0
    for j in range(1, N - 1):
        if j % 2 != 0:
            w[j] = 4.0

    # multiply
    F_in = G * w

    fft_vals = np.fft.fft(F_in)
    # scale => call(k_j)= e^{-alpha k_j}/ pi * real( fft_vals[j]) * lam / 3?
    # Because partial Simpson => sum_j= (delta/3)* ...
    # so we do an extra factor of ( lam/3 ) since lam= pi/B => delta= B/N => etc.

    # The factor is lam/3 because the spacing is lam in freq domain and we used a /3 from Simpson
    factor = lam / 3.0
    call_array = np.exp(-alpha * k_vals) / math.pi * np.real(fft_vals) * factor

    return call_array, k_vals


def fft_multi_strike_bsm(
    S0: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    alpha: float,
    N: int,
    B: float,
    strikes: List[float],
    is_call: bool = True,
    use_parallel: bool = False,
) -> List[float]:
    """
    Top-tier HPC function:
    1) Build normalized BSM CF => phi_X(u) for X= ln(S_T/S0).
    2) Do a single-lap domain [-B/2 .. +B/2].
    3) Interpolate for each strike in 'strikes'.
    4) If is_call=False => put-call parity => put = call - S0 e^{-qT}+K e^{-rT}.

    Returns
    -------
    List of option prices for each strike in the same order.
    """
    # Build CF
    phi_norm = bsm_char_func_normalized(r, q, sigma, T)
    # Single-lap
    call_arr, k_arr = cm_fft_shifted(phi_norm, r, T, alpha, N, B, use_parallel)

    # if we want to interpret each index j => K_j= S0 * exp(k_arr[j]).
    # Then we do interpolation for user strikes.

    # discount for put-call parity
    disc_s = math.exp(-q * T)
    disc_k = math.exp(-r * T)

    results = []
    for K in strikes:
        lnKoverS0 = math.log(K / S0)
        # find index => interpolation
        idx = np.searchsorted(k_arr, lnKoverS0)
        if idx <= 0:
            c_val = call_arr[0]
        elif idx >= len(k_arr):
            c_val = call_arr[-1]
        else:
            # linear interpolation
            k1 = k_arr[idx - 1]
            k2 = k_arr[idx]
            c1 = call_arr[idx - 1]
            c2 = call_arr[idx]
            c_val = c1 + (c2 - c1) * (lnKoverS0 - k1) / (k2 - k1)
        if is_call:
            results.append(float(c_val))
        else:
            # put= call - S0 e^{-qT}+ K e^{-rT}
            put_val = c_val - (S0 * disc_s) + (K * disc_k)
            results.append(float(put_val))

    return results


if __name__ == "__main__":
    """
    We'll do a test with S0=100, r=0.03, q=0.01, sigma=0.2, T=1.0,
    alpha=1.5, N=2**12, B=10.0 => domain k in [-5.. +5].
    Strikes=80..120 => we should see non-zero call prices.
    """
    alpha = 1.1
    N = 2**12
    B = 25.0  # covering e^(-5) ~ 0.0067 * S0 up to e^5 ~ 148.4 * S0
    use_parallel = True

    S0, r, q, sigma, T = 100.0, 0.03, 0.01, 0.2, 1.0
    test_strikes = [80.0, 90.0, 100.0, 110.0, 120.0]

    calls = fft_multi_strike_bsm(
        S0,
        r,
        q,
        sigma,
        T,
        alpha,
        N,
        B,
        test_strikes,
        is_call=True,
        use_parallel=use_parallel,
    )
    for k, px in zip(test_strikes, calls):
        print(f"Strike={k}, CallPrice={px:.4f}")
