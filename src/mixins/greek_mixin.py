import concurrent.futures
from functools import lru_cache
from typing import Any, Dict
from dataclasses import is_dataclass, replace


class GreekMixin:
    """
    Mixin that implements Greek calculations (delta, gamma, vega, theta, rho)
    via central finite differences, using the parent's .price(...) method.
    """

    def __init__(self, parallel: bool = False, disable_cache: bool = True) -> None:
        # Children call super() for multiple inheritance
        self._parallel_fd = parallel
        self.disable_cache = disable_cache

    @lru_cache(None)
    def _cached_price(
        self,
        instrument: Any,
        underlying: Any,
        model: Any,
        market_env: Any,
        **kwargs,
    ) -> float:
        """
        LRU-cached thin wrapper around self.price(...).
        """
        return self.price(instrument, underlying, model, market_env, **kwargs)

    def get_price(
        self,
        instrument: Any,
        underlying: Any,
        model: Any,
        market_env: Any,
        **kwargs,
    ) -> float:
        """
        Return the price, using the cache if not disabled.
        """
        if self.disable_cache:
            return self.price(instrument, underlying, model, market_env, **kwargs)
        else:
            return self._cached_price(
                instrument, underlying, model, market_env, **kwargs
            )

    def _finite_diff_1st(
        self,
        fn,
        base_args: Dict[str, Any],
        param_name: str,
        step: float,
    ) -> float:
        """
        Central difference for the first derivative w.r.t. param_name.
        """
        base_val = base_args[param_name]
        # Create parameter sets for x+h and x-h.
        up_args = dict(base_args)
        dn_args = dict(base_args)
        up_args[param_name] = base_val + step
        dn_args[param_name] = base_val - step

        if self._parallel_fd:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                f_up_future = executor.submit(fn, **up_args)
                f_dn_future = executor.submit(fn, **dn_args)
                f_up = f_up_future.result()
                f_dn = f_dn_future.result()
        else:
            f_up = fn(**up_args)
            f_dn = fn(**dn_args)
        return (f_up - f_dn) / (2.0 * step)

    def _finite_diff_2nd(
        self,
        fn,
        base_args: Dict[str, Any],
        param_name: str,
        step: float,
    ) -> float:
        """
        Central difference for the second derivative w.r.t. param_name.
        """
        base_val = base_args[param_name]
        up_args = dict(base_args)
        mid_args = dict(base_args)
        dn_args = dict(base_args)
        up_args[param_name] = base_val + step
        dn_args[param_name] = base_val - step

        if self._parallel_fd:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                f_up_future = executor.submit(fn, **up_args)
                f_mid_future = executor.submit(fn, **mid_args)
                f_dn_future = executor.submit(fn, **dn_args)
                f_up = f_up_future.result()
                f_mid = f_mid_future.result()
                f_dn = f_dn_future.result()
        else:
            f_up = fn(**up_args)
            f_mid = fn(**mid_args)
            f_dn = fn(**dn_args)
        return (f_up - 2.0 * f_mid + f_dn) / (step * step)

    def delta(
        self,
        instrument: Any,
        underlying: Any,
        model: Any,
        market_env: Any,
        step: float = None,
        **kwargs,
    ) -> float:
        """
        Delta via finite difference w.r.t. the underlying spot price.
        """
        spot = getattr(underlying, "spot", None)
        if spot is None or spot <= 0:
            raise ValueError("Underlying spot must be positive for delta FD.")

        if step is None:
            step = 1e-4 * spot  # default step is 0.01% of spot

        base_args = {
            "instrument": instrument,
            "underlying": underlying,
            "model": model,
            "market_env": market_env,
            "spot": spot,
        }

        def param_wrapper(**kwargs2):
            # Remove duplicate keys
            kwargs2.pop("instrument", None)
            kwargs2.pop("underlying", None)
            kwargs2.pop("model", None)
            kwargs2.pop("market_env", None)
            # Get the shifted parameter.
            param_spot = kwargs2.pop("spot", None)
            if param_spot is not None:
                underlying.spot = param_spot
            result = self.get_price(
                instrument, underlying, model, market_env, **kwargs2
            )
            underlying.spot = spot  # revert to original spot

            return result

        return self._finite_diff_1st(param_wrapper, base_args, "spot", step)

    def gamma(
        self,
        instrument: Any,
        underlying: Any,
        model: Any,
        market_env: Any,
        step: float = None,
        **kwargs,
    ) -> float:
        """
        Gamma via finite difference of the first derivative w.r.t. spot.
        """
        spot = getattr(underlying, "spot", None)
        if spot is None or spot <= 0:
            return 0.0
        if step is None:
            step = 1e-4 * spot

        base_args = {
            "instrument": instrument,
            "underlying": underlying,
            "model": model,
            "market_env": market_env,
            "spot": spot,
        }

        def param_wrapper(**kwargs2):
            # Remove duplicate keys
            kwargs2.pop("instrument", None)
            kwargs2.pop("underlying", None)
            kwargs2.pop("model", None)
            kwargs2.pop("market_env", None)
            param_spot = kwargs2.pop("spot", None)
            if param_spot is not None:
                underlying.spot = param_spot
            result = self.get_price(
                instrument, underlying, model, market_env, **kwargs2
            )
            underlying.spot = spot  # revert to original spot
            return result

        return self._finite_diff_2nd(param_wrapper, base_args, "spot", step)

    def vega(
        self,
        instrument: Any,
        underlying: Any,
        model: Any,
        market_env: Any,
        step: float = 1e-4,
        **kwargs,
    ) -> float:
        """
        Vega via finite difference w.r.t. the volatility.
        """
        sigma = getattr(underlying, "volatility", None)
        if sigma is None:
            sigma = getattr(model, "sigma", None)
        if sigma is None or sigma <= 0:
            return 0.0
        if step is None:
            step = 1e-4 * sigma

        base_args = {
            "instrument": instrument,
            "underlying": underlying,
            "model": model,
            "market_env": market_env,
            "sigma": sigma,
        }

        def param_wrapper(**kwargs2):
            # Remove duplicate keys
            kwargs2.pop("instrument", None)
            kwargs2.pop("underlying", None)
            kwargs2.pop("model", None)
            kwargs2.pop("market_env", None)
            param_sigma = kwargs2.pop("sigma", None)
            if hasattr(underlying, "volatility"):
                if param_sigma is not None:
                    underlying.volatility = param_sigma
                result = self.get_price(
                    instrument, underlying, model, market_env, **kwargs2
                )
                underlying.volatility = sigma  # revert
            else:
                old_sigma = getattr(model, "sigma", None)
                if param_sigma is not None:
                    model.sigma = param_sigma
                result = self.get_price(
                    instrument, underlying, model, market_env, **kwargs2
                )
                if old_sigma is not None:
                    model.sigma = old_sigma
            return result

        return self._finite_diff_1st(param_wrapper, base_args, "sigma", step)

    def theta(
        self,
        instrument: Any,
        underlying: Any,
        model: Any,
        market_env: Any,
        step: float = 1e-4,
        **kwargs,
    ) -> float:
        """
        Compute theta via finite difference with respect to maturity (time to expiry).

        Parameters
        ----------
        instrument : Any
            The option instrument. It should have a 'maturity' attribute.
        underlying : Any
            The underlying asset.
        model : Any
            The pricing model.
        market_env : Any
            The market environment.
        step : float, optional
            Step size for finite difference (default is 1e-4 or 0.1*T, whichever is smaller).
        **kwargs : dict
            Additional parameters passed to the pricing function.

        Returns
        -------
        float
            The computed theta.
        """
        T = getattr(instrument, "maturity", None)
        if T is None or T <= 0:
            return 0.0
        if step is None:
            step = min(1e-4, 0.1 * T)

        base_args = {
            "instrument": instrument,
            "underlying": underlying,
            "model": model,
            "market_env": market_env,
            "maturity": T,
        }

        def param_wrapper(**kwargs2):
            # Remove duplicate keys.
            kwargs2.pop("instrument", None)
            kwargs2.pop("underlying", None)
            kwargs2.pop("model", None)
            kwargs2.pop("market_env", None)
            param_T = kwargs2.pop("maturity", None)
            if param_T is not None:
                # If instrument is a dataclass instance, use replace.
                if is_dataclass(instrument):
                    new_instrument = replace(instrument, maturity=param_T)
                # If not a dataclass but supports attribute assignment, update temporarily.
                elif hasattr(instrument, "__dict__"):
                    orig = instrument.maturity
                    instrument.maturity = param_T
                    new_instrument = instrument
                else:
                    raise TypeError(
                        "Instrument does not support attribute update or is not a dataclass"
                    )
            else:
                new_instrument = instrument

            result = self.get_price(
                new_instrument, underlying, model, market_env, **kwargs2
            )

            # If the instrument was modified temporarily, revert the change.
            if (
                not is_dataclass(instrument)
                and hasattr(instrument, "__dict__")
                and param_T is not None
            ):
                instrument.maturity = orig

            return result

        return -self._finite_diff_1st(param_wrapper, base_args, "maturity", step)

    def rho(
        self,
        instrument: Any,
        underlying: Any,
        model: Any,
        market_env: Any,
        step: float = 1e-4,
        **kwargs,
    ) -> float:
        """
        Rho via finite difference w.r.t. risk-free rate.
        """
        r = getattr(market_env, "rate", None)
        if r is None:
            r = getattr(model, "r", None)
        if r is None:
            raise ValueError("Cannot find interest rate for Rho calculation.")

        base_args = {
            "instrument": instrument,
            "underlying": underlying,
            "model": model,
            "market_env": market_env,
            "rate": r,
        }

        def param_wrapper(**kwargs2):
            # Remove duplicate keys
            kwargs2.pop("instrument", None)
            kwargs2.pop("underlying", None)
            kwargs2.pop("model", None)
            kwargs2.pop("market_env", None)
            param_r = kwargs2.pop("rate", None)
            if param_r is not None and hasattr(market_env, "rate"):
                market_env.rate = param_r
                result = self.get_price(
                    instrument, underlying, model, market_env, **kwargs2
                )
                market_env.rate = r  # revert to original rate
            else:
                old_r = getattr(model, "r", 0.0)
                if param_r is not None:
                    model.r = param_r
                result = self.get_price(
                    instrument, underlying, model, market_env, **kwargs2
                )
                model.r = old_r
            return result

        return self._finite_diff_1st(param_wrapper, base_args, "rate", step)
