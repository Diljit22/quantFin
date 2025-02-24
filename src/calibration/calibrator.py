import numpy as np
from scipy.optimize import minimize
import logging
from concurrent.futures import ProcessPoolExecutor


class Calibrator:
    """
    A generic calibrator for financial models.

    The model should implement:
      - update_params(params: dict) to update its calibration parameters.
      - price_option(S, K, T, r, q) to compute the option price.

    The calibrator minimizes the squared error between model and market prices.
    """

    def __init__(
        self,
        model,
        market_data,
        model_params_keys,
        initial_guess,
        bounds=None,
        options=None,
        use_parallel=False,
        n_jobs=1,
    ):
        """
        Parameters:
            model: An instance of a pricing model.
            market_data: Dictionary containing:
                - strikes: numpy array of strikes.
                - maturities: numpy array of maturities.
                - market_prices: numpy array of observed option prices.
                - S: underlying asset price.
                - r: risk-free rate.
                - q: dividend yield.
            model_params_keys: List of parameter names to calibrate.
            initial_guess: Initial guess for parameters (list or numpy array).
            bounds: List of (min, max) tuples for each parameter.
            options: Dictionary of options for the optimizer.
            use_parallel: If True, evaluate option prices in parallel.
            n_jobs: Number of parallel workers.
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
    def _evaluate_option_price(args):
        """
        Evaluate an option price in a separate process.

        Args:
            args: Tuple (K, T, model_params, S, r, q, model_class)
        """
        K, T, model_params, S, r, q, model_class = args
        model_copy = model_class(**model_params)
        return model_copy.price_option(S, K, T, r, q)

    def objective_function(self, params):
        """
        Objective function: sum of squared differences between model prices and market prices.
        """
        # Update the model parameters.
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

        Returns:
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
