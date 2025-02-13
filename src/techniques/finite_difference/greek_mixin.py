import math
import copy
import concurrent.futures
from functools import lru_cache
from typing import Dict, Any, Optional

import numpy as np

from src.techniques.base_technique import BaseTechnique
from src.instruments.base_instrument import BaseInstrument
from src.models.base_model import BaseModel
from src.market.market_environment import MarketEnvironment


# ------------------------------------------------------------------------
# 1) GreekMixin for Finite Differences
# ------------------------------------------------------------------------
class GreekMixin:
    """
    Mixin providing finite-difference methods for Greeks and general
    sensitivity calculations. The actual pricing function (or method)
    must be provided by the class that inherits this mixin.

    Attributes
    ----------
    parallel : bool
        If True, use multithreading for each finite-difference perturbation.
        May help in HPC contexts if pricing calls are CPU-bound.

    Methods
    -------
    cached_price(**params) -> float:
        A cached version of the core pricing routine. Must be overridden
        or provided by the child class. We typically define `_price(...)`
        in the child and call it from this method.

    finite_difference_greek(param_name, base_params, step, order=1) -> float:
        Central finite-difference approximation for a single derivative
        wrt one parameter.

    multi_parameter_diff(...) -> Optional[Dict[str, float]]:
        A more advanced interface if you want to simultaneously compute
        partial derivatives with respect to multiple parameters.
    """

    def __init__(self, parallel: bool = False):
        self.use_parallel = parallel
        super().__init__()

    @lru_cache(maxsize=None)
    def cached_price(self, **params) -> float:
        """
        Cached price function. By default, calls `_price(...)`, which
        must be implemented by the inheriting class.

        Note: The dictionary 'params' must be hashable for lru_cache
        to work properly. One trick is to convert all mutable structures
        to tuples or strings.
        """
        # The child class (FiniteDiffTechnique) will implement `_price(...)`.
        return self._price(**params)

    def finite_difference_greek(
        self, param_name: str, base_params: Dict[str, Any], step: float, order: int = 1
    ) -> float:
        """
        Compute a central finite-difference derivative of the specified
        order w.r.t. one parameter.

        Parameters
        ----------
        param_name : str
            The parameter key in 'base_params' to differentiate w.r.t.
        base_params : Dict[str, Any]
            A dictionary of (param_name -> value) used for pricing.
        step : float
            The finite-difference step size for the selected parameter.
        order : int, default=1
            Which derivative we want (1=first derivative, 2=second derivative).

        Returns
        -------
        float
            Approximate partial derivative of the given order.
        """
        if param_name not in base_params:
            raise ValueError(f"Parameter '{param_name}' not found in base_params.")

        # For HPC, if order=1, we do 2 calls; if order=2, we do 3 calls.
        # We can parallelize these calls if self.parallel=True.
        base_value = base_params[param_name]

        if order == 1:
            # d/d(param_name)
            # central difference: f(x+h) - f(x-h)) / (2h)
            param_sets = [
                (param_name, base_value + step),
                (param_name, base_value - step),
            ]
        elif order == 2:
            # second derivative: (f(x+h) - 2f(x) + f(x-h)) / h^2
            param_sets = [
                (param_name, base_value + step),
                (param_name, base_value),
                (param_name, base_value - step),
            ]
        else:
            raise NotImplementedError(
                "Only up to second derivative is implemented here."
            )

        # Build the partial functions to call
        def _worker(shifted_value):
            new_params = dict(base_params)
            new_params[param_name] = shifted_value
            # Convert to a hashable signature for caching
            # e.g., convert dict -> tuple sorted by key
            # but lru_cache might handle the hashing if we are careful
            # We'll do an ad-hoc approach:
            signature = tuple(sorted(new_params.items()))
            return self.cached_price(**dict(signature))

        if self.use_parallel and len(param_sets) > 1:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(_worker, val) for _, val in param_sets]
                results = [f.result() for f in futures]
        else:
            results = [_worker(val) for _, val in param_sets]

        if order == 1:
            # results = [f(x+h), f(x-h)]
            diff_val = (results[0] - results[1]) / (2.0 * step)
        else:
            # results = [f(x+h), f(x), f(x-h)]
            diff_val = (results[0] - 2.0 * results[1] + results[2]) / (step**2)

        return diff_val

    def multi_parameter_diff(
        self, base_params: Dict[str, Any], param_steps: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Example of simultaneously computing partial derivatives w.r.t.
        multiple parameters. For demonstration only. Possibly HPC.

        Returns a dict: { param_name : derivative_val }.

        We do a first-order central difference for each param in param_steps.
        """
        results = {}

        # If many parameters, we can parallelize each derivative if desired.
        if self.use_parallel and len(param_steps) > 1:

            def _compute_single_greek(param_name: str, step: float):  # -> (str, float):
                val = self.finite_difference_greek(
                    param_name, base_params, step, order=1
                )
                return (param_name, val)

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(_compute_single_greek, p, s)
                    for p, s in param_steps.items()
                ]
                for f in futures:
                    p_name, val = f.result()
                    results[p_name] = val
        else:
            for p, s in param_steps.items():
                results[p] = self.finite_difference_greek(p, base_params, s, order=1)

        return results

    # The child class will define `_price(...)`.
