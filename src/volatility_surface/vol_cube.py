"""
vol_cube.py
===========
A universal "vol cube" (or vol surface) builder that computes
implied volatilities across a 2D grid of (strike, maturity) pairs
using any technique conforming to BaseTechnique.

Key features:
-------------
1. Allows specifying many (strike, maturity) combos + their target prices.
2. Uses technique.implied_volatility(...) to get IV for each combo.
3. Optionally runs in parallel with ThreadPoolExecutor.
4. Stores final results in a Pandas DataFrame with columns:
    - strike
    - maturity
    - market_price
    - implied_vol
    - model_technique_label
    - option_type

Usage Example:
--------------
>>> from vol_cube import VolCube
>>> from src.instruments.european_option import EuropeanOption
>>> from src.underlyings.stock import Stock
>>> from src.market.market_enviroment import MarketEnvironment
>>> from src.models.black_scholes_merton import BlackScholesMerton
>>> from src.techniques.closed_forms.bsm_technique import BlackScholesMertonTechnique
>>>
>>> strikes = [90, 100, 110]
>>> maturities = [0.5, 1.0, 2.0]
>>> market_prices = {
...    (90, 0.5): 12.45,
...    (90, 1.0): 14.10,
...    (90, 2.0): 17.80,
...    (100, 0.5): 7.20,
...    (100, 1.0): 9.50,
...    (100, 2.0): 13.75,
...    (110, 0.5): 3.30,
...    (110, 1.0): 5.80,
...    (110, 2.0): 9.50
... }
>>>
>>> technique = BlackScholesMertonTechnique(cache_results=True)
>>> underlying = Stock(spot=100.0, volatility=0.2, dividend=0.01)
>>> market_env = MarketEnvironment(rate=0.03)
>>> model = BlackScholesMerton(sigma=0.2)
>>>
>>> vcube = VolCube(
...     technique=technique,
...     underlying=underlying,
...     model=model,
...     market_env=market_env,
...     is_call=True
... )
>>>
>>> df = vcube.build(
...     strikes=strikes,
...     maturities=maturities,
...     market_prices=market_prices,
...     parallel=True,
...     max_workers=4,
...     tol=1e-6,
...     max_iter=100
... )
>>>
>>> print(df)
# MultiIndex DataFrame of implied vols + label, etc.

"""

import math
import concurrent.futures
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any


class VolCube:
    """
    A "VolCube" or vol surface builder that, for a given technique + underlying + model
    + market environment, computes implied volatilities across a 2D grid of strikes
    and maturities, using provided market prices.

    Parameters
    ----------
    technique : BaseTechnique
        Any technique implementing .implied_volatility(...)
    underlying : Any
        Underlying object used by the technique (e.g., Stock).
    model : Any
        Model object used by the technique (e.g. BlackScholesMerton).
    market_env : Any
        Market environment object used by the technique (e.g. MarketEnvironment).
    is_call : bool, optional (default=True)
        Whether we are computing implied vols for call options or puts.

    Attributes
    ----------
    technique : BaseTechnique
    underlying : Any
    model : Any
    market_env : Any
    is_call : bool
    """

    def __init__(
        self,
        technique: Any,
        underlying: Any,
        model: Any,
        market_env: Any,
        is_call: bool = True,
    ):
        self.technique = technique
        self.underlying = underlying
        self.model = model
        self.market_env = market_env
        self.is_call = is_call

        # Attempt to capture some descriptive names for labeling
        self.model_name = getattr(model, "model_name", str(model.__class__.__name__))
        self.technique_name = self.technique.__class__.__name__
        self.model_technique_label = f"{self.model_name} ({self.technique_name})"

    def build(
        self,
        strikes: List[float],
        maturities: List[float],
        market_prices: Dict[Tuple[float, float], float],
        parallel: bool = False,
        max_workers: int = 4,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Build an implied vol "surface" (DataFrame) for the given grid of (strike, maturity).

        Parameters
        ----------
        strikes : List[float]
            A list of strike prices.
        maturities : List[float]
            A list of maturities (in years).
        market_prices : Dict[(float, float), float]
            A dict mapping (strike, maturity) -> observed market price.
        parallel : bool, optional
            If True, compute implied vols in parallel via ThreadPoolExecutor.
        max_workers : int, optional
            Number of threads if parallel=True (default=4).
        **kwargs :
            Additional arguments to pass to technique.implied_volatility(...)
            (e.g., tol=1e-7, max_iter=100, initial_guess=0.2, etc.)

        Returns
        -------
        pd.DataFrame
            A DataFrame with a MultiIndex of (strike, maturity), containing:
                implied_vol,
                market_price,
                model_technique_label,
                option_type,
            as columns.

        Raises
        ------
        ValueError
            If any (strike, maturity) in the grid is missing from market_prices.
        RuntimeError
            If the implied volatility calculation fails for a given pair
            and we choose to raise an exception (otherwise we might store NaN).
        """

        # Prepare a list of tasks
        combos = []
        for K in strikes:
            for T in maturities:
                if (K, T) not in market_prices:
                    raise ValueError(
                        f"Missing market price for (strike={K}, maturity={T})."
                    )
                px = market_prices[(K, T)]
                combos.append((K, T, px))

        # Dispatch either parallel or serial
        if parallel:
            results = self._compute_parallel(combos, max_workers, **kwargs)
        else:
            results = []
            for K, T, px in combos:
                iv = self._compute_iv_for_combo(K, T, px, **kwargs)
                results.append((K, T, px, iv))

        # Create a DataFrame
        df = pd.DataFrame(
            results, columns=["strike", "maturity", "market_price", "implied_vol"]
        )

        # Add columns for labeling
        df["model_technique_label"] = self.model_technique_label
        df["option_type"] = "Call" if self.is_call else "Put"

        # Set MultiIndex
        df.set_index(["strike", "maturity"], inplace=True)
        return df

    def _compute_parallel(
        self,
        combos: List[Tuple[float, float, float]],
        max_workers: int,
        **kwargs,
    ) -> List[Tuple[float, float, float, float]]:
        """
        Parallel computation of implied vols for each (K, T, px).
        Returns a list of (K, T, px, iv).
        """
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {}
            for K, T, px in combos:
                fut = executor.submit(self._compute_iv_for_combo, K, T, px, **kwargs)
                future_map[fut] = (K, T, px)

            for fut in concurrent.futures.as_completed(future_map):
                K, T, px = future_map[fut]
                try:
                    iv = fut.result()
                except Exception as e:
                    # If we want to fail on the first error, re-raise:
                    raise RuntimeError(f"IV calc failed for (K={K}, T={T}).") from e
                results.append((K, T, px, iv))
        return results

    def _compute_iv_for_combo(
        self,
        strike: float,
        maturity: float,
        price: float,
        **kwargs,
    ) -> float:
        """
        Helper that constructs an option object and calls .implied_volatility(...).
        If it fails, you can either return NaN or raise an error.
        """

        # You presumably have a class EuropeanOption, or you might build from your base instrument class
        # Example:
        from src.instruments.european_option import EuropeanOption

        option = EuropeanOption(
            strike=strike,
            maturity=maturity,
            is_call=self.is_call,
        )

        # Now call the technique's implied_vol method
        iv = self.technique.implied_volatility(
            instrument=option,
            underlying=self.underlying,
            model=self.model,
            market_env=self.market_env,
            target_price=price,
            **kwargs,
        )
        return iv
