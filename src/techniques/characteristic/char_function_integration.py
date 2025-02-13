# src/techniques/characteristic/char_function_integration.py

"""
CharFunctionIntegrationTechnique
================================

Implements a direct characteristic function integration approach
for European options, using scipy.integrate.quad from 0..∞.

Key Points
----------
- Inherits from FiniteDiffTechnique so that it can reuse the finite-diff
  logic for Greeks (gamma, vega, theta, rho) and implied vol.
- Overrides _price(...) so the parent’s finite-diff calls see this integral
  logic, not a fallback snippet.
- The user-facing price(...) method also calls the same integral approach
  for convenience.

References
----------
Often referred to as a "second approach" snippet. The integrals:
  pITMCall = 0.5 + (1/pi) * ∫[0..∞] Im( e^{-i*u*ln(K)} * φ(u) ) / u du
  deltaCall= 0.5 + (1/pi) * ∫[0..∞] Im( e^{-i*u*ln(K)} * φ(u - i) ) / u du
  call = (S e^{-qT})*deltaCall - (K e^{-rT})*pITMCall
  put  = call - S e^{-qT} + K e^{-rT}

Usage
-----
>>> from src.techniques.characteristic.char_function_integration import CharFunctionIntegrationTechnique
>>> tech = CharFunctionIntegrationTechnique()
>>> px = tech.price(my_option, my_bsm_model, my_market_env)
>>> # greeks like gamma => uses the parent's finite-diff
>>> gam = tech.gamma(my_option, my_bsm_model, my_market_env)
print("Gamma:", gam)
"""

import math
import cmath
import numpy as np
import scipy.integrate
from typing import Any

from src.techniques.finite_difference.finite_difference import FiniteDiffTechnique
from src.instruments.base_instrument import BaseInstrument
from src.models.base_model import BaseModel
from src.market.market_environment import MarketEnvironment


class CharFunctionIntegrationTechnique(FiniteDiffTechnique):
    """
    A technique class that uses characteristic-function integration
    for European calls/puts. Inherits FiniteDiffTechnique so that
    delta/gamma/vega/etc. are computed by finite-difference unless
    specifically overridden.

    Notes
    -----
    - HPC concurrency is omitted here for simplicity. If needed, adapt
      chunk-based integration or parallel frameworks.
    - We assume the model defines model.characteristic_function(...),
      which we call with arguments (T, u=..., spot=S, r=r, q=q).
    """

    def __init__(self) -> None:
        """
        We skip parallel or extra HPC parameters for now.
        Just call the parent with use_parallel=False.
        """
        super().__init__(use_parallel=False)
        self.name = "CharFuncIntegration"

    def price(
        self,
        instrument: BaseInstrument,
        model: BaseModel,
        market_env: MarketEnvironment,
        **kwargs
    ) -> float:
        """
        User-facing method: compute the option price from
        (instrument, model, market_env) directly.

        We parse the required data and call our integral approach.

        Parameters
        ----------
        instrument : BaseInstrument
            Must be a European call or put with a strike, maturity, etc.
        model : BaseModel
            Must define characteristic_function(t, u=..., spot, r, q).
        market_env : MarketEnvironment
            Provides the current spot, etc.
        **kwargs : dict
            Currently unused.

        Returns
        -------
        float
            The option price computed by characteristic-function integrals.
        """
        if instrument.option_style.lower() != "european":
            raise ValueError("CharFunctionIntegrationTechnique only supports European.")

        S = market_env.get_spot_price(instrument.underlying_symbol)
        K = instrument.strike
        if K is None:
            raise ValueError("Instrument must define a strike for pricing.")
        T = instrument.maturity
        is_call = instrument.option_type.lower() == "call"

        # Extract model parameters
        m_params = model.get_params()
        r = m_params.get("r", 0.0)
        q = m_params.get("dividend_yield", 0.0)

        # Perform integral approach
        return self._char_func_integration(S, K, T, r, q, is_call, model)

    # ----------------------------------------------------------------
    # OVERRIDE: _price(...) so parent's finite-diff calls come here
    # ----------------------------------------------------------------
    def _price(self, **params) -> float:
        """
        Internal method used by FiniteDiffTechnique for repeated calls
        during Greek computations or implied vol root-finding.

        We parse the param dictionary: {spot, strike, maturity, r, q, sigma, option_type, __model__}
        then run the same integral logic.

        Returns
        -------
        float
            The option price under these parameters.
        """
        spot = params["spot"]
        strike = params["strike"]
        T = params["maturity"]
        r = params["r"]
        q = params["q"]
        sigma = params["sigma"]
        otype = params["option_type"]
        model = params.get("__model__", None)

        if model is None:
            raise ValueError(
                "No model object provided in param dictionary for CharFuncIntegration."
            )

        # We assume the model's 'sigma' is overwritten or the characteristic function
        # logic references model._params["sigma"].
        # So let's ensure we do: model._params["sigma"] = sigma
        model._params["sigma"] = sigma

        is_call = otype.lower() == "call"

        # do the integral approach
        return self._char_func_integration(spot, strike, T, r, q, is_call, model)

    # ----------------------------------------------------------------
    # Actual characteristic function integral approach
    # ----------------------------------------------------------------

    def _char_func_integration(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        is_call: bool,
        model: BaseModel,
        only_delta: bool = False,
    ) -> float:
        """
        Implementation of the "second approach" integrals:
          pITMCall = 0.5 + (1/pi)* ∫ Im( e^{-i*u*ln(K)} * φ(u) ) / u   du
          deltaCall= 0.5 + (1/pi)* ∫ Im( e^{-i*u*ln(K)} * φ(u - i) ) / u du
          call     = (S e^{-qT}) * deltaCall - (K e^{-rT}) * pITMCall
          put      = call - S e^{-qT} + K e^{-rT}

        For T near zero, we do an intrinsic payoff.

        Parameters
        ----------
        S : float
            Current spot price
        K : float
            Strike
        T : float
            Time to maturity
        r : float
            Risk-free rate
        q : float
            Continuous dividend yield
        is_call : bool
            True => call, False => put
        model : BaseModel
            Must define characteristic_function(t, u=..., spot=S, r=r, q=q).

        Returns
        -------
        float
            Option price from char-function integration.
        """
        if T <= 1e-12:
            payoff = (S - K) if is_call else (K - S)
            return max(0.0, payoff)

        cf = model.char_func(S, T)
        if not cf:
            raise NotImplementedError("Model missing characteristic_function(...)")

        lnK = math.log(K)

        twPhi = lambda u: cf(u - 1j) / cf(-1j)
        trfPhi = lambda u: np.imag(np.exp(-1j * u * lnK) * cf(u)) / u
        trfTwi = lambda u: np.imag(np.exp(-1j * u * lnK) * twPhi(u)) / u

        A, _ = scipy.integrate.quad(trfPhi, 0, np.inf)
        B, _ = scipy.integrate.quad(trfTwi, 0, np.inf)

        pITMCall = 0.5 + A / math.pi
        deltaCall = 0.5 + B / math.pi

        if only_delta:
            return deltaCall if is_call else deltaCall - 1.0

        disc_s = S * math.exp(-q * T)
        disc_k = K * math.exp(-r * T)

        if is_call:
            call_px = disc_s * deltaCall - disc_k * pITMCall
            return call_px
        else:
            # put = call - S e^{-qT} + K e^{-rT}
            put_px = disc_k * (1 - pITMCall) - disc_s * (1 - deltaCall)
            return put_px

    def delta(
        self,
        instrument: BaseInstrument,
        model: BaseModel,
        market_env: MarketEnvironment,
        **kwargs
    ) -> float:
        """
        User-facing method: compute the option price from
        (instrument, model, market_env) directly.

        We parse the required data and call our integral approach.

        Parameters
        ----------
        instrument : BaseInstrument
            Must be a European call or put with a strike, maturity, etc.
        model : BaseModel
            Must define characteristic_function(t, u=..., spot, r, q).
        market_env : MarketEnvironment
            Provides the current spot, etc.
        **kwargs : dict
            Currently unused.

        Returns
        -------
        float
            The option price computed by characteristic-function integrals.
        """
        if instrument.option_style.lower() != "european":
            raise ValueError("CharFunctionIntegrationTechnique only supports European.")

        S = market_env.get_spot_price(instrument.underlying_symbol)
        K = instrument.strike
        if K is None:
            raise ValueError("Instrument must define a strike for pricing.")
        T = instrument.maturity
        is_call = instrument.option_type.lower() == "call"

        # Extract model parameters
        m_params = model.get_params()
        r = m_params.get("r", 0.0)
        q = m_params.get("dividend_yield", 0.0)

        # Perform integral approach
        return self._char_func_integration(
            S, K, T, r, q, is_call, model, only_delta=True
        )
