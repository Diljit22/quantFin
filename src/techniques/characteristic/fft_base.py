# -*- coding: utf-8 -*-
"""
Carr-Madan FFT-based European Option Pricer using the prFFT approach with debugging
and optional normalization (S normalized to 1). This version fixes the log–strike grid
by centering it at the reference strike and applies an FFT shift to align the FFT output.

References:
- Carr, P., and Madan, D. B. (1999). "Option valuation using the fast Fourier transform."
"""

import numpy as np
import math
import multiprocessing

# --------------------------------------------------------------------------
# Supporting functions (from your prFFT code)
# --------------------------------------------------------------------------


def isATM(S, K, eps=0.01):
    """Return True if the option is at-the-money (|S-K| <= eps)."""
    return abs(S - K) <= eps


def genFuncs(phi, S, K, alpha, disc, eps=0.01):
    """
    Generate functions required for FFT pricing.

    Returns:
      dampener : function to damp the integrand (ensures square-integrability)
      twiPhi   : transformed characteristic function used in the Fourier transform.
    """
    atmFlag = isATM(S, K, eps)
    if atmFlag:
        dampener = lambda x: np.exp(alpha * x)
        denom_ = lambda u: alpha**2 + alpha - u**2 + 1j * u * (2 * alpha + 1)
        twiPhi = lambda u: phi(u - 1j * (1 + alpha)) / denom_(u)
    else:
        dampener = lambda x: np.sinh(alpha * x)
        ft = (
            lambda u: 1 / (1j * u + 1)
            - 1 / (disc * 1j * u)
            - phi(u - 1j) / (u**2 - 1j * u)
        )
        twiPhi = lambda u: (ft(u - 1j * alpha) - ft(u + 1j * alpha)) / 2
    return dampener, twiPhi


# --------------------------------------------------------------------------
# HPCFFTOptionPricer Class using prFFT method with normalization & debug output
# --------------------------------------------------------------------------


class HPCFFTOptionPricer:
    """
    Carr-Madan FFT pricer using the prFFT approach.

    This implementation does not use numba and is designed for multi-core CPUs.
    """

    def __init__(self, alpha=1.0, r=0.01, T=1.0):
        """
        Initialize the pricer.

        Parameters:
          alpha : Dampening parameter.
          r     : Risk-free interest rate.
          T     : Time to maturity.
        """
        self.alpha = float(alpha)
        self.r = float(r)
        self.T = float(T)

    def price_calls_fft(
        self,
        char_func,
        S,
        K,
        q=0.0,
        call=True,
        ATMeps=0.01,
        trunc=7,
        n=12,
        strikes=None,
        normalize=False,
        debug=False,
    ):
        """
        Price European options via FFT (Carr-Madan / prFFT approach).

        Parameters:
          char_func : callable
              The characteristic function φ(u) of the log–price.
          S         : float
              Underlying asset price.
          K         : float
              Reference strike price (the FFT will be centered so that k_ref = log(K)).
          q         : float
              Continuous dividend yield.
          call      : bool
              True for call option; False for put.
          ATMeps    : float
              Tolerance for determining if the option is at–the–money.
          trunc     : int
              Exponent for the truncation limit (B = 2**trunc).
          n         : int
              Determines the number of FFT points L = 2**n.
          strikes   : np.ndarray or None
              If provided, a 1D array of strike prices at which to interpolate
              the FFT–computed prices. Otherwise, returns the full grid and the
              option price at the reference strike.
          normalize : bool
              If True, the underlying is normalized (S is set to 1 and K becomes 1)
              and the final results are rescaled by the original S.
          debug     : bool
              If True, print intermediate computation values.

        Returns:
          If strikes is provided:
              (prices_interp, (K_grid, values))
          Otherwise:
              (price_at_reference, (K_grid, values))

        Notes:
          If normalization is used, the provided char_func should be defined for S = 1.
        """
        # Save original S for rescaling later if normalization is used.
        S_orig = S
        if normalize:
            if debug:
                print("Normalizing: Original S =", S_orig)
            S = 1.0
            # Also normalize strike so that at-the-money means K=1.
            K = K / S_orig
            if debug:
                print("Normalized strike K =", K)

        # Compute reference log–strike (should be the center of the FFT grid)
        k_ref = np.log(K)
        disc = math.exp(-self.r * self.T)
        if debug:
            print("k_ref (log(K)) =", k_ref)
            print("Discount factor disc =", disc)

        # Generate dampening and transformed characteristic function
        dampener, twi = genFuncs(char_func, S, K, self.alpha, disc, eps=ATMeps)
        damp_kref = dampener(k_ref)
        if debug:
            print("dampener(k_ref) =", damp_kref)

        # Set up the Fourier integration grid
        dy = 2 ** (trunc - n)  # Step size in Fourier domain
        B = 2**trunc  # Upper limit of integration in Y-space
        L = 2**n  # Number of FFT points

        if debug:
            print("Integration grid parameters:")
            print("  dy =", dy)
            print("  B  =", B)
            print("  L  =", L)

        # Scaling multiplier (from prFFT function)
        mul = disc * (B / math.pi) / damp_kref
        if debug:
            print("Scaling multiplier (mul) =", mul)

        # Integration variable grid (Y-space)
        Y = np.arange(0, B, dy)  # Should yield L points
        if debug:
            print("First 5 values of Y =", Y[:5])

        # Compute transformed integrand Q (include the integration step dy)
        Q = np.exp(-1j * k_ref * Y) * twi(Y)
        Q[0] /= 2  # Adjust the first term
        if debug:
            print("First 5 values of Q =", Q[:5])

        # Compute inverse FFT to get option prices (in "price–space")
        fft_result = np.fft.ifft(Q)
        # Multiply by scaling multiplier and then shift the output so that the center
        # of the FFT corresponds to k_ref.
        values = mul * np.real(fft_result)
        values = np.fft.fftshift(values)
        if debug:
            print("First 5 FFT output values after fftshift =", values[:5])

        # Construct the log–strike grid centered at k_ref.
        delta_k = 2 * math.pi / B
        k_grid = k_ref + (np.arange(L) - L / 2) * delta_k
        K_grid = np.exp(k_grid)
        if debug:
            print("delta_k =", delta_k)
            print("First 5 values of k_grid =", k_grid[:5])
            print("First 5 values of K_grid (before rescaling) =", K_grid[:5])

        # If normalized, rescale the strike grid back to the original scale.
        if normalize:
            K_grid = S_orig * K_grid
            if debug:
                print("Rescaled K_grid (first 5) =", K_grid[:5])

        # For a put, adjust via put–call parity.
        if not call:
            values = values - S * math.exp(-q * self.T) + K * disc

        # If a specific strike array is provided, interpolate.
        if strikes is not None:
            prices_interp = np.interp(strikes, K_grid, values)
            if normalize:
                prices_interp = S_orig * prices_interp
            return prices_interp, (K_grid, values)
        else:
            # Find the index where K_grid is closest to the reference strike.
            ref_target = K if not normalize else S_orig * K
            idx = np.argmin(np.abs(K_grid - ref_target))
            price_ref = values[idx]
            if normalize:
                price_ref *= S_orig
            return price_ref, (K_grid, values)


# --------------------------------------------------------------------------
# Example Usage
# --------------------------------------------------------------------------


def example_usage():
    """
    Example usage of HPCFFTOptionPricer with a Black-Scholes characteristic function.

    We run the FFT pricer both without normalization and with normalization.
    """
    # Model parameters
    S0 = 100.0  # Underlying price
    sigma = 0.2  # Volatility
    r = 0.01  # Risk-free rate
    T = 1.0  # Time to maturity
    q = 0.0  # Dividend yield

    # Define Black-Scholes characteristic function.
    # For the non–normalized case, include log(S0).
    def bs_char_func_full(u):
        i = 1j
        mu = np.log(S0) + (r - 0.5 * sigma**2) * T
        return np.exp(i * u * mu - 0.5 * sigma**2 * T * u**2)

    # For the normalized case (S=1), use log(S)=0.
    def bs_char_func_norm(u):
        i = 1j
        mu = (r - q - 0.5 * sigma**2) * T  # with S=1, log(1)=0.
        return np.exp(i * u * mu - 0.5 * sigma**2 * T * u**2)

    # Instantiate the pricer with alpha=1.0.
    pricer = HPCFFTOptionPricer(alpha=1.0, r=r, T=T)

    # Set a reference strike.
    ref_strike = 100.0  # Price an option with strike 100.
    interp_strikes = np.array([90, 95, 100, 105, 110], dtype=np.float32)

    print("\n=== Without Normalization ===")
    price_no_norm, (K_grid_no_norm, values_no_norm) = pricer.price_calls_fft(
        char_func=bs_char_func_full,
        S=S0,
        K=ref_strike,
        q=q,
        call=True,
        ATMeps=0.01,
        trunc=7,
        n=12,
        strikes=None,
        normalize=False,
        debug=True,
    )
    print("Option Price at Reference Strike (K=100):", price_no_norm)

    prices_interp_no_norm, _ = pricer.price_calls_fft(
        char_func=bs_char_func_full,
        S=S0,
        K=ref_strike,
        q=q,
        call=True,
        ATMeps=0.01,
        trunc=7,
        n=12,
        strikes=interp_strikes,
        normalize=False,
        debug=False,
    )
    print("Input strikes:", interp_strikes)
    print("Interpolated Option Prices:", prices_interp_no_norm)

    print("\n=== With Normalization (S normalized to 1) ===")
    price_norm, (K_grid_norm, values_norm) = pricer.price_calls_fft(
        char_func=bs_char_func_norm,
        S=S0,
        K=ref_strike,
        q=q,
        call=True,
        ATMeps=0.01,
        trunc=7,
        n=12,
        strikes=None,
        normalize=True,
        debug=True,
    )
    print("Option Price at Reference Strike (K=100):", price_norm)

    prices_interp_norm, _ = pricer.price_calls_fft(
        char_func=bs_char_func_norm,
        S=S0,
        K=ref_strike,
        q=q,
        call=True,
        ATMeps=0.01,
        trunc=7,
        n=12,
        strikes=interp_strikes,
        normalize=True,
        debug=False,
    )
    print("Input strikes:", interp_strikes)
    print("Interpolated Option Prices:", prices_interp_norm)


def test_fft():
    """
    Example usage of HPCFFTOptionPricer with a Black-Scholes characteristic function.

    We run the FFT pricer both without normalization and with normalization.
    """
    # Model parameters
    S0 = 100.0  # Underlying price
    sigma = 0.2  # Volatility
    r = 0.01  # Risk-free rate
    T = 1.0  # Time to maturity
    q = 0.0  # Dividend yield

    def bs_char_func_full(u):
        i = 1j
        mu = np.log(S0) + (r - 0.5 * sigma**2) * T
        return np.exp(i * u * mu - 0.5 * sigma**2 * T * u**2)

    # For the normalized case (S=1), use log(S)=0.
    def bs_char_func_norm(u):
        i = 1j
        mu = (r - q - 0.5 * sigma**2) * T  # with S=1, log(1)=0.
        return np.exp(i * u * mu - 0.5 * sigma**2 * T * u**2)

    # Instantiate the pricer with alpha=1.0.
    pricer = HPCFFTOptionPricer(alpha=1.0, r=r, T=T)

    # Set a reference strike.
    ref_strike = 100.0  # Price an option with strike 100.
    interp_strikes = np.array([90, 95, 100, 105, 110], dtype=np.float32)

    price_no_norm, (K_grid_no_norm, values_no_norm) = pricer.price_calls_fft(
        char_func=bs_char_func_full,
        S=S0,
        K=ref_strike,
        q=q,
        call=True,
        ATMeps=0.01,
        trunc=7,
        n=12,
        strikes=None,
        normalize=False,
        debug=False,
    )
    print("Option Price at Reference Strike (K=100):", price_no_norm)

    prices_interp_no_norm, _ = pricer.price_calls_fft(
        char_func=bs_char_func_full,
        S=S0,
        K=ref_strike,
        q=q,
        call=True,
        ATMeps=0.01,
        trunc=7,
        n=12,
        strikes=interp_strikes,
        normalize=False,
        debug=False,
    )
    for i, a in enumerate(interp_strikes):

        print(f"{a}        {prices_interp_no_norm[i]}")


if __name__ == "__main__":
    test_fft()
