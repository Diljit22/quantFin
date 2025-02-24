#!/usr/bin/env python3
"""
data_downloader.py

This module provides the DataDownloader class to fetch historical data from FRED and Polygon.
It downloads:
  - Historical stock closing prices (from Polygon).
  - Option chain data (actual option prices; here simulated for demonstration).
  - Risk‑free rates from FRED via yield curve interpolation.

Downloaded data are also logged to CSV files for auditing.
"""

import csv
import logging
import os
import numpy as np
from datetime import datetime

from src.live.polygon_option_data import PolygonAPIClient
from src.live.fred_long_term_rate import (
    FredAPIClient,
    get_yield_curve,
    interpolate_rate,
)

# Configure logging.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


class DataDownloader:
    def __init__(
        self, polygon_api_key: str, fred_api_key: str, data_dir: str = "data_logs"
    ):
        self.polygon_api_key = polygon_api_key
        self.fred_api_key = fred_api_key
        self.polygon_client = PolygonAPIClient(api_key=polygon_api_key)
        self.fred_client = FredAPIClient(fred_api_key)
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def download_stock_data(self, stock_symbol: str, from_date: str, to_date: str):
        """
        Download historical stock closing prices for the given date range.
        Returns a list of closing prices.
        """
        prices = self.polygon_client.get_stock_historical_prices(
            stock_symbol, from_date, to_date
        )
        filename = os.path.join(
            self.data_dir, f"{stock_symbol}_stock_data_{from_date}_to_{to_date}.csv"
        )
        with open(filename, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Index", "ClosingPrice"])
            for i, price in enumerate(prices):
                writer.writerow([i, price])
        logger.info(
            "Downloaded and logged stock data for %s from %s to %s",
            stock_symbol,
            from_date,
            to_date,
        )
        return prices

    def download_option_chain(
        self, stock_symbol: str, date: str, option_type: str = "C"
    ):
        """
        Download the option chain (actual option prices) for the given stock on a given date.
        For demonstration, this returns simulated data.
        Returns a list of dictionaries with 'strike' and 'price'.
        """
        # Simulate option chain: strikes from 400 to 500 (step 5).
        strikes = np.arange(400, 501, 5)
        # For simulation, assume a dummy underlying spot of 450.
        spot = 450.0
        option_chain = []
        for strike in strikes:
            intrinsic = max(spot - strike, 0)
            noise = np.random.normal(0, 0.5)
            price = max(intrinsic + noise, 0)
            option_chain.append({"strike": float(strike), "price": price})
        filename = os.path.join(
            self.data_dir, f"{stock_symbol}_option_chain_{date}.csv"
        )
        with open(filename, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Strike", "ActualPrice"])
            for entry in option_chain:
                writer.writerow([entry["strike"], entry["price"]])
        logger.info(
            "Downloaded and logged option chain for %s on %s", stock_symbol, date
        )
        return option_chain

    def download_risk_free_rate(self, target_maturity: float) -> float:
        """
        Download the risk‑free rate from FRED for the given target maturity.
        Returns the rate as a decimal.
        """
        series_map = {
            1: "DGS1",
            2: "DGS2",
            3: "DGS3",
            5: "DGS5",
            7: "DGS7",
            10: "DGS10",
            20: "DGS20",
            30: "DGS30",
        }
        yield_curve = get_yield_curve(self.fred_client, series_map)
        risk_free_rate_percent = interpolate_rate(target_maturity, yield_curve)
        rf_rate = risk_free_rate_percent / 100.0
        filename = os.path.join(self.data_dir, "risk_free_rates.csv")
        with open(filename, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().isoformat(), target_maturity, rf_rate])
        logger.info(
            "Downloaded risk‑free rate for maturity %.2f: %.4f",
            target_maturity,
            rf_rate,
        )
        return rf_rate
