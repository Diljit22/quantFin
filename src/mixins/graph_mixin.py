"""
graph_mixin.py
==============

Provides a Mixin that offers a default `graph(...)` method for pricing
techniques. The method varies a specified numeric parameter of the instrument
(default is "strike") and computes the price at each point, plotting a
curve of parameter vs. price.

Usage
-----
- Inherit from `GraphMixin` alongside your base technique class.
- Ensure `self.price(...)` is implemented; `graph(...)` calls it internally.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Optional, Tuple


class GraphMixin:
    """
    Mixin that provides a default graphing method for a pricing technique.

    The method `graph(...)` varies a specified numerical parameter of the instrument
    (default is "strike") and computes prices, then plots parameter vs. price.
    """

    def graph(
        self,
        instrument: Any,
        underlying: Any,
        model: Any,
        market_env: Any,
        param_name: str = "strike",
        num_points: int = 50,
        param_range: Optional[Tuple[float, float]] = None,
    ) -> None:
        """
        Default graphing method for pricing techniques.

        Varies the specified numerical parameter (e.g., "strike") of the
        instrument and computes price for each value, then plots parameter vs. price.

        Parameters
        ----------
        parameter : str, optional
            The name of the numerical parameter to vary (default "strike").
        num_points : int, optional
            Number of points for the curve (default 50).
        param_range : tuple of (float, float), optional
            (min, max) range for the parameter. If not provided,
            defaults to (0.5 * current_value, 1.5 * current_value).

        Returns
        -------
        None
            Displays a Matplotlib plot of parameter vs. price.
        """
        container_dict = {"strike": instrument}
        if param_name not in container_dict:
            raise AttributeError(
                f"Instrument has no attribute '{param_name}'. "
                "Please ensure the instrument has this numeric field."
            )
        param_container = container_dict[param_name]
        current_value = getattr(param_container, param_name, None)

        if param_range is None:
            lower = 0.5 * current_value
            upper = 1.5 * current_value
        else:
            lower, upper = param_range

        param_values = np.linspace(lower, upper, num_points)
        prices = []

        for value in param_values:
            adj_instrument = instrument.with_strike(value)
            # Compute the price using the child's .price(...)
            price_val = self.price(adj_instrument, underlying, model, market_env)

            prices.append(price_val)

        # Plot the parameter vs. price
        plt.figure(figsize=(10, 6))
        plt.plot(param_values, prices, marker="o", linestyle="-")
        plt.xlabel(param_name.capitalize())
        plt.ylabel("Price")
        plt.title(f"Price vs. {param_name.capitalize()}")
        plt.grid(True)
        plt.show()
