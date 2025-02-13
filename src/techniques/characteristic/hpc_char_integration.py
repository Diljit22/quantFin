import math
import cmath
from networkx import sigma
import numpy as np
from typing import Any, Dict
import concurrent.futures

from src.techniques.base_technique import BaseTechnique
from src.techniques.finite_difference.finite_difference import FiniteDiffTechnique
from src.instruments.base_instrument import BaseInstrument
from src.models.base_model import BaseModel
from src.market.market_environment import MarketEnvironment

import math
import cmath
import numpy as np
import scipy.integrate
from typing import Any

from src.techniques.base_technique import BaseTechnique
from src.techniques.finite_difference.finite_difference import FiniteDiffTechnique
from src.instruments.base_instrument import BaseInstrument
from src.models.base_model import BaseModel
from src.market.market_environment import MarketEnvironment


class CharFunctionIntegrationTechnique2(FiniteDiffTechnique):
    """
    Implements a direct characteristic function integration approach
    for European options. Subclasses FiniteDiffTechnique to reuse its
    finite-difference Greeks and implied vol logic.

    By default, price(...) will do the characteristic-function integration
    for a single strike. The Greeks (delta, gamma, etc.) fallback to
    finite-difference (unless you override them with direct formulas).

    If the user wants to handle multiple strikes at once for HPC,
    a specialized method could be added to compute a grid in parallel.

    Parameters
    ----------
    num_points : int, optional
        Number of grid points for the numeric integral in [0, u_max].
        The integrand typically decays for large u, so choose carefully.
    u_max : float, optional
        Upper limit for the integration domain (0..∞ truncated to 0..u_max).
    use_parallel : bool, optional
        If True, attempt to parallelize the integral evaluation
        in sub-intervals for HPC usage.
    """

    def __init__(
        self, num_points: int = 2000, u_max: float = 100.0, use_parallel: bool = False
    ) -> None:
        """
        Constructor for the characteristic function integration technique.

        Attributes
        ----------
        num_points : int
            Discretization for the numerical integration.
        u_max : float
            Upper bound to truncate the integral at. For stable integrands,
            100.0 is often enough, but depends on model tail behavior.
        use_parallel : bool
            If True, attempts concurrency in the integration.
        """
        super().__init__(use_parallel=use_parallel)
        self.num_points = num_points
        self.u_max = u_max

    # ----------------------------------------------------------------
    # Override price(...) to do characteristic function integration
    # ----------------------------------------------------------------

    def price(
        self, instrument: BaseInstrument, model: BaseModel, market_env: Any, **kwargs
    ) -> float:
        """
        Price a European call/put by numerical integration of
        the characteristic function. For HPC, we chunk the domain
        if parallel is requested.

        Parameters
        ----------
        instrument : BaseInstrument
            Typically a European Option with strike, maturity, option_type.
        model : BaseModel
            Must implement characteristic_function(t, u=complex, spot=..., r=..., q=...).
        market_env : MarketEnvironment
            Provides the spot price (and possibly data).
        **kwargs : dict
            Additional parameters for advanced usage (like parallel chunk sizes).

        Returns
        -------
        float
            The computed option price.
        """
        # We can handle only European for now
        if getattr(instrument, "option_style", "european").lower() != "european":
            raise ValueError(
                "CharFunctionIntegrationTechnique only supports European options."
            )

        # Extract data
        S = market_env.get_spot_price(instrument.underlying_symbol)
        strike = getattr(instrument, "strike", None)
        if strike is None:
            raise ValueError("Instrument must have a strike.")
        T = instrument.maturity
        is_call = getattr(instrument, "option_type", "call").lower() == "call"

        # Pull model params
        params = model.get_params()
        r = params.get("r", 0.0)
        q = params.get("dividend_yield", 0.0)

        # Get the characteristic function from the model
        cf = getattr(model, "characteristic_function", None)
        if cf is None:
            # fallback or error
            raise NotImplementedError(
                "Model must define characteristic_function for char-function integration."
            )

        # Price a call with two integrals: p1, p2
        # p1 = 0.5 + 1/pi ∫[0..∞] ...
        # p2 = 0.5 + 1/pi ∫[0..∞] ...
        # Then call = S e^{-qT} * p1 - K e^{-rT} * p2
        # For put => put = call - S e^{-qT} + K e^{-rT}

        lnK = math.log(strike)
        discount_q = math.exp(-q * T)
        discount_r = math.exp(-r * T)

        # We'll define the integrand functions:

        def integrand_p1(u: float) -> float:
            # p1 integrand => Im( e^{-i u ln(K)} * cf(T, u=... ) / (i u) )
            # We pass 'u' as real, but the CF needs a complex argument.
            # Specifically, we pass u (which is real) -> u is purely imaginary exponent?
            u_cplx = complex(u, 0.0)
            val_cf = cf(T, u=u_cplx, spot=S, r=r, q=q)
            # e^{-i u lnK}
            e_term = cmath.exp(complex(0, -u) * lnK)
            # denom => i * u => complex(0, u)
            # so dividing by i u => multiply by -i / u if we handle carefully.
            # But let's just do the imaginary part directly.
            numerator = e_term * val_cf
            # We want Im( numerator / (i u) ) = Im( numerator * [-i / u] )
            # multiply by -i => multiply by complex(0, -1)
            # So let's define something:
            #  comp = numerator * complex(0, -1) / u
            #  result = comp.real  (since multiplying by i rotates by 90°, so we want real part)
            comp = numerator * complex(0, -1) / u
            return (
                comp.real
            )  # real part corresponds to the imaginary part of original expression

        def integrand_p2(u: float) -> float:
            # p2 integrand => Im( e^{-i u lnK} * cf(T, u=(u - i),...) / (i u) )
            # so we shift the argument by -i =>  (u - i) => complex(u, -1)
            u_shifted = complex(u, -1)
            val_cf = cf(T, u=u_shifted, spot=S, r=r, q=q)
            e_term = cmath.exp(complex(0, -u) * lnK)
            comp = e_term * val_cf * complex(0, -1) / u
            return comp.real

        # Now numerically integrate each from 0..∞ (approx 0..u_max)
        # We'll do trapz with self.num_points in [0..self.u_max].

        def chunked_integration(
            fun, u_min: float, u_max: float, n_points: int
        ) -> float:
            """
            Trapz integration of 'fun' from u_min to u_max with n_points spacing.
            If self.use_parallel is True, chunk into subintervals.
            """
            eps = 1e-8
            if u_min < eps:
                u_min = eps
            if not self.use_parallel:
                # single-chunk approach
                us = np.linspace(u_min, u_max, n_points)
                vals = np.array([fun(u) for u in us])
                return np.trapz(vals, us)
            else:
                # multi-chunk approach
                # e.g., split the domain into 4 subintervals
                # Could param as well
                n_chunks = 4
                chunk_size = n_points // n_chunks
                chunk_ranges = []
                step = (u_max - u_min) / n_chunks
                for i in range(n_chunks):
                    start = u_min + i * step
                    end = u_min + (i + 1) * step
                    n_sub = (
                        chunk_size
                        if i < n_chunks - 1
                        else (n_points - chunk_size * (n_chunks - 1))
                    )
                    chunk_ranges.append((start, end, n_sub))

                # We'll integrate each chunk in parallel
                def do_chunk(start, end, n_sub):
                    sub_us = np.linspace(start, end, n_sub)
                    sub_vals = np.array([fun(xx) for xx in sub_us])
                    return np.trapz(sub_vals, sub_us)

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [
                        executor.submit(do_chunk, s, e, nn)
                        for (s, e, nn) in chunk_ranges
                    ]
                    partials = [f.result() for f in futures]
                return sum(partials)

        # compute p1
        integral_p1 = chunked_integration(
            integrand_p1, 0.0, self.u_max, self.num_points
        )
        p1 = 0.5 + (1.0 / math.pi) * integral_p1

        # compute p2
        integral_p2 = chunked_integration(
            integrand_p2, 0.0, self.u_max, self.num_points
        )
        p2 = 0.5 + (1.0 / math.pi) * integral_p2

        call_price = (S * discount_q * p1) - (strike * discount_r * p2)
        if is_call:
            return float(call_price)
        else:
            # put = call - S e^{-qT} + K e^{-rT}
            put_price = call_price - (S * discount_q) + (strike * discount_r)
            return float(put_price)

    @property
    def name(self):
        return "CharFunc Integr2"
