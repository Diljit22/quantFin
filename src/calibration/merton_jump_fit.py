#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
merton_jump_fit.py
==================
Calibration framework for the MertonJump model.

This module defines a generic Calibrator class to calibrate financial models.
The model to be calibrated must implement:
  - update_params(params: dict): update its calibration parameters.
  - price_option(S, K, T, r, q): compute the option price.

The calibrator minimizes the squared error between model and market prices.
"""

import numpy as np
from scipy.optimize import minimize
import logging
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Any, List

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


class Calibrator:
    """
    A generic calibrator for financial models.

    The model should implement:
      - update_params(params: dict) to update its calibration parameters.
      - price_option(S, K, T, r, q) to compute the option price.

    The calibrator minimizes the squared error between model prices and market prices.
    """

    def __init__(
        self,
        model: Any,
        market_data: Dict[str, Any],
        model_params_keys: List[str],
        initial_guess,
        bounds: List[tuple] = None,
        options: Dict[str, Any] = None,
        use_parallel: bool = False,
        n_jobs: int = 1,
    ):
        """
        Parameters
        ----------
        model : object
            An instance of a pricing model.
        market_data : dict
            Dictionary containing:
              - strikes: numpy array of strikes.
              - maturities: numpy array of maturities.
              - market_prices: numpy array of observed option prices.
              - S: underlying asset price.
              - r: risk-free rate.
              - q: dividend yield.
        model_params_keys : list of str
            List of parameter names to calibrate.
        initial_guess : array-like
            Initial guess for the parameters.
        bounds : list of tuple, optional
            Bounds for each parameter.
        options : dict, optional
            Options for the optimizer.
        use_parallel : bool, optional
            If True, evaluate option prices in parallel.
        n_jobs : int, optional
            Number of parallel workers.
        """
        self.model = model
        self.market_data = market_data
        self.param_keys = model_params_keys
        self.initial_guess = np.array(initial_guess)
        self.bounds = bounds
        self.options = (
            options if options is not None else {"disp": True, "maxiter": 500}
        )
        self.use_parallel = use_parallel
        self.n_jobs = n_jobs
        self.logger = logging.getLogger(__name__)
        self.result = None

    @staticmethod
    def _evaluate_option_price(args) -> float:
        """
        Evaluate an option price in a separate process.

        Parameters
        ----------
        args : tuple
            A tuple (K, T, model_params, S, r, q, model_class).

        Returns
        -------
        float
            The option price computed by a copy of the model.
        """
        K, T, model_params, S, r, q, model_class = args
        model_copy = model_class(**model_params)
        return model_copy.price_option(S, K, T, r, q)

    def objective_function(self, params) -> float:
        """
        Objective function: Sum of squared differences between model and market prices.

        Parameters
        ----------
        params : array-like
            Array of model parameter values.

        Returns
        -------
        float
            Sum of squared errors between model prices and market prices.
        """
        param_dict = dict(zip(self.param_keys, params))
        self.model.update_params(param_dict)
        strikes = self.market_data["strikes"]
        maturities = self.market_data["maturities"]
        S = self.market_data["S"]
        r = self.market_data["r"]
        q = self.market_data["q"]

        if self.use_parallel:
            tasks = [
                (K, T, param_dict, S, r, q, self.model.__class__)
                for K, T in zip(strikes, maturities)
            ]
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                model_prices = list(executor.map(self._evaluate_option_price, tasks))
        else:
            model_prices = [
                self.model.price_option(S, K, T, r, q)
                for K, T in zip(strikes, maturities)
            ]

        market_prices = self.market_data["market_prices"]
        error = np.array(model_prices) - market_prices
        sq_error = np.sum(error**2)
        self.logger.debug("Params: %s, Squared Error: %f", params, sq_error)
        return sq_error

    def calibrate(self):
        """
        Run the calibration process.

        Returns
        -------
        OptimizeResult
            The result of the optimization.
        """
        self.logger.info(
            "Starting calibration with initial guess: %s", self.initial_guess
        )
        result = minimize(
            self.objective_function,
            self.initial_guess,
            bounds=self.bounds,
            options=self.options,
        )
        self.result = result
        if result.success:
            self.logger.info("Calibration successful: %s", result)
        else:
            self.logger.error("Calibration failed: %s", result)
        return result
