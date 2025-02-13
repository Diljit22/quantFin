"""
BaseModel
=========

A model, in this context, encapsulates the 
mathematical assumptions about the evolution of an underlying asset 
(e.g., Black-Scholes, Heston, Variance Gamma), including any relevant 
parameters (volatility, mean reversion, jump intensities, etc.).

Design & Usage
--------------
- Subclasses are expected to provide model-specific details:
  * SDE definitions (drift, diffusion).
  * Characteristic functions (for FFT-based methods).
  * Path simulation routines (for Monte Carlo).
- By default, this base class defines a minimal interface:
  * A method `price(...)` that may be optionally implemented for closed-form
    solutions directly in the model. In many designs, the actual pricing is
    delegated to a separate 'Technique' object (e.g., PDE, MC, etc.).
  * A method `get_params()` for retrieving the model's parameters.
  * Optional placeholders (e.g., `characteristic_function(...)`, `simulate_paths(...)`)
    that raise `NotImplementedError` unless overridden.

Performance & HPC
-----------------
- Derived models that compute path simulations or characteristic functions 
  are good candidates for:
  * Vectorization (NumPy arrays) 
  * Parallelization (e.g., `multiprocessing` or `joblib`)
  * Just-In-Time compilation (Numba) or Cython for tight loops
- For real-time ensure thread-safety when 
  sharing stateful objects (like rate curves) and be mindful of concurrency 
  control if needed.

Examples
--------
Implementing a simple Black-Scholes model subclass might look like:

>>> class BlackScholesModel(BaseModel):
...     def __init__(self, sigma: float, r: float, dividend_yield: float = 0.0):
...         super().__init__(model_name="BlackScholes", sigma=sigma, r=r,
...                          dividend_yield=dividend_yield)
...
...     def drift(self, t: float, s: float) -> float:
...         return (self._params['r'] - self._params['dividend_yield']) * s
...
...     def diffusion(self, t: float, s: float) -> float:
...         return self._params['sigma'] * s
...
...     def price(self, *args, **kwargs) -> float:
...         # Possibly call a closed-form formula or pass to a technique.
...         # For demonstration, just raise NotImplementedError
...         raise NotImplementedError("Use a technique or override with closed-form.")
"""

import abc
from typing import Any, Dict, Optional


class BaseModel(abc.ABC):
    """
    Abstract base class for all quantitative models.

    Each model encapsulates a set of assumptions/parameters about how
    an underlying asset evolves over time, optionally providing:
    - SDE functions: drift, diffusion (if relevant).
    - Characteristic functions (for FFT-based pricing).
    - Path simulation routines (for Monte Carlo).
    - Direct or closed-form pricing methods (if applicable).

    Attributes
    ----------
    _model_name : str
        A short descriptive name (e.g., "BlackScholes", "Heston").
    _params : Dict[str, Any]
        A dictionary of model-specific parameters (e.g., volatility, mean reversion).
    """

    def __init__(self, model_name: str = "GenericModel", **model_params: Any) -> None:
        """
        Initialize the base model with a name and arbitrary keyword parameters.

        Parameters
        ----------
        model_name : str, optional
            A descriptive name or identifier for the model.
        **model_params : dict
            Arbitrary keyword arguments representing model parameters.

        Notes
        -----
        - Validation of model parameters should typically be done
          in subclasses or via the `validate_params()` method.
        """
        self._model_name = model_name
        self._params = dict(model_params)
        self.validate_params()

    @property
    def model_name(self) -> str:
        """
        str : A short descriptive name for this model.
        """
        return self._model_name

    def get_params(self) -> Dict[str, Any]:
        """
        Retrieve the model's parameters as a dictionary.

        Returns
        -------
        Dict[str, Any]
            A copy of the internal dictionary containing model parameters.

        Notes
        -----
        - This method allows pricing techniques or other library components
          to query the model for needed parameters.
        """
        return dict(self._params)

    def validate_params(self) -> None:
        """
        Validate the model parameters.

        Raises
        ------
        ValueError
            If any model parameters are invalid.

        Notes
        -----
        - This method is intended to be overridden by subclasses
          to ensure that parameters (e.g., volatility >= 0) meet
          the model's constraints.
        - The default implementation does no validation.
        """
        pass

    @abc.abstractmethod
    def price(self, *args, **kwargs) -> float:
        """
        (Optional) Compute the fair value of an instrument under this model.

        This method may or may not be relevant depending on the design:
        - In some architectures, the 'Technique' class is responsible for
          pricing based on the model's parameters (i.e., the model doesn't
          price on its own).
        - In others, closed-form solutions can be placed here directly.

        Parameters
        ----------
        *args :
            Positional arguments relevant to the pricing routine.
        **kwargs :
            Keyword arguments relevant to the pricing routine.

        Returns
        -------
        float
            The fair value computed by this model.

        Raises
        ------
        NotImplementedError
            If a subclass does not implement this and instead relies
            exclusively on external techniques.
        """
        pass

    @abc.abstractmethod
    def get_stochastic_process(self, *args, **kwargs):  # -> BaseStochasticProcess:
        """
        Return a stochastic process (SDE) object that encodes this model's
        dynamics (e.g., drift, diffusion).

        Returns
        -------
        BaseStochasticProcess
            An instance of a stochastic process class implementing the model's SDE.

        Raises
        ------
        NotImplementedError
            If the model does not define a corresponding stochastic process.
        """
        pass

    def characteristic_function(self, t: float, *args, **kwargs) -> complex:
        """
        Compute the characteristic function of the log-price under this model
        at time t.

        Parameters
        ----------
        t : float
            Time parameter (e.g., time to maturity).
        *args :
            Additional positional arguments.
        **kwargs :
            Additional keyword arguments.

        Returns
        -------
        complex
            The value of the characteristic function at time t.

        Raises
        ------
        NotImplementedError
            If the model does not define a characteristic function.
        """
        raise NotImplementedError(
            f"{self.model_name} does not define a characteristic function."
        )

    def pde_params(self, S: float, t: float = 0.0, **kwargs) -> Dict[str, float]:
        """
        Retrieve PDE coefficients for a standard (1D) Black–Scholes PDE
        at a given spot S and time t. Typically returns:
          - 'diffusion': sigma^2
          - 'drift': (r - q)
          - 'rate': r
          - (any extra data PDE solvers might need)

        Parameters
        ----------
        S : float
            Current underlying price or grid point. By default, some PDE
            approaches may not strictly require this if volatility is constant.
        t : float, default=0.0
            Current time, in years, for time-dependent PDE or rates if needed.
        **kwargs :
            Additional PDE-related parameters (unused by default).

        Returns
        -------
        dict
            A dictionary of PDE coefficients. For example:
                {
                  "diffusion": <float>,
                  "drift": <float>,
                  "rate": <float>
                }

        Raises
        ------
        NotImplementedError
            If not implemented by a subclass.
        """
        raise NotImplementedError(f"{self.model_name} does not define PDE parameters.")

    def simulate_paths(self, n_paths: int, n_steps: int, *args, **kwargs) -> Any:
        """
        Simulate sample paths under this model's stochastic process.

        Parameters
        ----------
        n_paths : int
            Number of independent paths to simulate.
        n_steps : int
            Number of time steps per path.
        *args :
            Additional positional arguments for the simulation method.
        **kwargs :
            Additional keyword arguments for the simulation method.

        Returns
        -------
        Any
            Typically a NumPy array or similar structure of shape (n_paths, n_steps)
            containing simulated paths.

        Raises
        ------
        NotImplementedError
            If the model does not define a path simulation method.
        """
        raise NotImplementedError(
            f"{self.model_name} does not implement path simulation."
        )

    def __repr__(self) -> str:
        """
        String representation for debugging.

        Returns
        -------
        str
            The model name and parameters in a human-readable format.
        """
        params_str = ", ".join(f"{k}={v!r}" for k, v in self._params.items())
        return (
            f"{self.__class__.__name__}(model_name={self._model_name!r}, {params_str})"
        )
