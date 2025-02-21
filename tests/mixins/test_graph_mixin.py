# test_graph_mixin.py

import numpy as np
import matplotlib.pyplot as plt
import pytest

from src.mixins.graph_mixin import GraphMixin


class DummyTechnique(GraphMixin):
    def price(self, instrument: any, underlying, model: any, market_env: any) -> float:
        """
        A dummy price function that returns 2 * instrument.strike + 1.
        """
        # Assumed the instrument has an attribute 'strike'
        return 2 * instrument.strike + 1


class DummyInstrument:
    def __init__(self, strike: float):
        self.strike = strike

    def with_strike(self, new_strike):
        self.strike = new_strike
        return self


def test_graph_mixin_linearity():
    """
    Test the GraphMixin.graph method by verifying that it plots a linear curve
    according to the dummy price function: price = 2 * strike + 1.
    """
    current_strike = 100.0
    dummy_instr = DummyInstrument(strike=current_strike)
    dummy_underlying = object()
    dummy_model = object()
    dummy_market_env = object()

    technique = DummyTechnique()

    param_range = (50, 150)
    num_points = 5

    # Monkey-patch plt.show to avoid blocking during tests.
    original_show = plt.show
    try:
        plt.show = lambda: None
        technique.graph(
            instrument=dummy_instr,
            underlying=dummy_underlying,
            model=dummy_model,
            market_env=dummy_market_env,
            param_name="strike",
            num_points=num_points,
            param_range=param_range,
        )

        fig = plt.gcf()
        ax = plt.gca()
        lines = ax.get_lines()
        # There should be one line.
        assert len(lines) == 1

        xdata = lines[0].get_xdata()
        ydata = lines[0].get_ydata()

        # Expected x-values: linearly spaced between 50 and 150 (num_points=5)
        expected_x = np.linspace(50, 150, num_points)
        # Expected y-values: using price = 2 * strike + 1.
        expected_y = 2 * expected_x + 1

        np.testing.assert_allclose(xdata, expected_x, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(ydata, expected_y, rtol=1e-5, atol=1e-8)
    finally:
        plt.show = original_show

    plt.clf()


def test_graph_mixin_missing_attribute():
    """
    Test that the graph method raises an AttributeError if the instrument
    does not have the specified numeric parameter.
    """
    dummy_instr = DummyInstrument(strike=100.0)
    dummy_underlying = object()
    dummy_model = object()
    dummy_market_env = object()
    technique = DummyTechnique()

    with pytest.raises(AttributeError):
        # Try to graph on a non-existent parameter 'maturity'
        technique.graph(
            instrument=dummy_instr,
            underlying=dummy_underlying,
            model=dummy_model,
            market_env=dummy_market_env,
            param_name="maturity",
            num_points=5,
        )
