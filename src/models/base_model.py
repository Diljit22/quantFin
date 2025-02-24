#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
base_model.py
=============

Abstract base class for all quantitative models. Each model encapsulates a set
of assumptions/parameters about how an underlying asset evolves over time.
"""

import abc
from typing import Any, Dict, Callable


class BaseModel(abc.ABC):
    """
    Abstract base class for all quantitative models.

    Each model encapsulates a set of assumptions/parameters about how
    an underlying asset evolves over time.

    Attributes
    ----------
    _model_name : str
        A descriptive name (e.g., "BlackScholesMerton", "Heston").
    _params : Dict[str, Any]
        A dictionary of model-specific parameters (e.g., volatility, mean reversion).
    """

    def __init__(self, model_name: str = "GenericModel", **model_params: Any) -> None:
        """
        Initialize the base model with a name and arbitrary keyword parameters.

        Parameters
        ----------
        model_name : str, optional
            A descriptive name or identifier for the model (default "GenericModel").
        **model_params : dict
            Arbitrary keyword arguments representing model parameters.
        """
        self._model_name = model_name
        self._params = dict(model_params)
        self.validate_params()

    @property
    def model_name(self) -> str:
        """
        str : Descriptive name for this model.
        """
        return self._model_name

    def get_params(self) -> Dict[str, Any]:
        """
        Retrieve the model's parameters as a dictionary.

        Returns
        -------
        Dict[str, Any]
            A copy of the internal dictionary containing model parameters.
        """
        return dict(self._params)

    def with_volatility(self, new_vol: float) -> "BaseModel":
        """
        Return a new instance of the model with the volatility parameter updated
        to new_vol.

        The model is assumed to store volatility under the key 'sigma' in its
        internal _params dictionary. A new instance of the same class is created
        with updated parameters.

        Parameters
        ----------
        new_vol : float
            The new volatility value.

        Returns
        -------
        BaseModel
            A new instance of the model with the updated volatility.

        Raises
        ------
        ValueError
            If the model does not have a 'sigma' parameter.
        """
        if "sigma" not in self._params:
            raise ValueError("This model does not have a 'sigma' parameter to update.")
        new_params = self.get_params()
        new_params["sigma"] = new_vol

        return self.__class__(**new_params)

    def validate_params(self) -> None:
        """
        Validate the model parameters.

        Raises
        ------
        ValueError
            If any model parameters are invalid.

        Notes
        -----
        Subclasses should override this if they have constraints on parameters.
        """
        pass

    def calibrate(self, market_data: Dict[str, Any]) -> None:
        """
        Calibrate the model parameters to market data.

        The calibration process minimizes the squared error between model and
        market prices. Subclasses should override this method if calibration
        is supported.

        Parameters
        ----------
        market_data : Dict[str, Any]
            A dictionary containing market data (e.g., option prices, strikes, maturities)
            against which to calibrate the model.

        Raises
        ------
        NotImplementedError
            Unless overridden by the subclass.
        """
        raise NotImplementedError(f"{self.model_name} does not implement calibration.")

    def SDE(self, *args, **kwargs) -> None:
        """
        Return or define a stochastic differential equation (SDE) object
        that encodes this model's dynamics.

        Raises
        ------
        NotImplementedError
            Unless overridden by a concrete model.
        """
        raise NotImplementedError(f"{self.model_name} has no SDE implementation.")

    def S_t(self, *args, **kwargs) -> None:
        """
        Return a function or object representing the stochastic process S(t).

        Raises
        ------
        NotImplementedError
            Unless overridden by a concrete model.
        """
        raise NotImplementedError(f"{self.model_name} has no S_t implementation.")

    def characteristic_function(self, *args, **kwargs) -> Callable[[complex], complex]:
        """
        Return a function phi(u) that computes the characteristic function
        of the log-price under this model at time t.

        Returns
        -------
        Callable[[complex], complex]
            A function phi(u) which, given a complex number u, returns
            the characteristic function value phi(u).
        """
        raise NotImplementedError(f"{self.model_name} has no char_func implementation.")

    def __repr__(self) -> str:
        """
        String representation of the model for debugging.

        Returns
        -------
        str
            A concise representation including model_name and parameters.
        """
        params_str = ", ".join(f"{k}={v!r}" for k, v in self._params.items())
        return (
            f"{self.__class__.__name__}(model_name={self._model_name!r}, {params_str})"
        )
