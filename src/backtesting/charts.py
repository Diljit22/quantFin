#!/usr/bin/env python3
"""
charts.py

This module reads performance metrics logged during backtesting from a CSV file and
produces charts to visualize the performance of your pricing model over time.
It uses matplotlib and pandas for charting.
"""

import pandas as pd
import matplotlib.pyplot as plt


def plot_performance_metrics(csv_file: str, output_file: str = "performance_chart.png"):
    """
    Reads a CSV file containing performance metrics (with columns Date, MAE, MSE)
    and plots the MAE and MSE over time.

    Parameters
    ----------
    csv_file : str
        Path to the CSV file containing performance metrics.
    output_file : str, optional
        Path to save the generated chart image. Defaults to "performance_chart.png".
    """
    # Read the CSV file into a DataFrame.
    df = pd.read_csv(csv_file)

    # Ensure that the 'Date' column is parsed as datetime.
    df["Date"] = pd.to_datetime(df["Date"])

    plt.figure(figsize=(10, 6))
    plt.plot(df["Date"], df["MAE"], label="MAE", marker="o", color="b")
    plt.plot(df["Date"], df["MSE"], label="MSE", marker="o", color="r")
    plt.xlabel("Date")
    plt.ylabel("Error")
    plt.title("Pricing Model Performance Over Time")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()


if __name__ == "__main__":
    # For testing purposes, change the CSV file path if necessary.
    csv_path = "performance_metrics.csv"
    plot_performance_metrics(csv_path)
