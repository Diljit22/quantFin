#!/usr/bin/env python3
"""
performance_evaluator.py

This module provides functions to evaluate the performance of pricing models
by comparing model prices with actual market prices.

The main function, evaluate_performance, computes metrics such as Mean Absolute Error (MAE)
and Mean Squared Error (MSE).
"""

import numpy as np


def evaluate_performance(priced_options):
    """
    Evaluate the performance of a pricing model.

    Parameters
    ----------
    priced_options : list of dict
         Each dict contains 'strike', 'actual_price', and 'model_price'.

    Returns
    -------
    dict
         Dictionary containing performance metrics (MAE, MSE).
    """
    actual = np.array([entry["actual_price"] for entry in priced_options])
    model = np.array([entry["model_price"] for entry in priced_options])
    mae = np.mean(np.abs(actual - model))
    mse = np.mean((actual - model) ** 2)
    return {"MAE": mae, "MSE": mse}
