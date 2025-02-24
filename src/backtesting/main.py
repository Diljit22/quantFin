#!/usr/bin/env python3
"""
main.py

This is the main script for backtesting option pricing models.
It uses the DataDownloader to retrieve historical data (stock data, option chains, risk‑free rates),
then prices options using a vectorized pricing function and evaluates performance metrics.
It also logs the execution time for pricing.
Usage:
    python main.py --stock SPY --start-date 2025-01-01 --end-date 2025-01-31 --option-type C
"""

import argparse
from datetime import datetime, timedelta
import csv
import os
import logging
import time  # <-- Import time module for timing
import numpy as np

from backtesting.data_downloader import DataDownloader
from backtesting.option_pricer import price_options_for_stock
from backtesting.performance_evaluator import evaluate_performance
from backtesting.pricing_functions import black_scholes_call_vectorized
from src.stock import Stock
from src.market_environment import MarketEnvironment
from evaluation.charts import plot_performance_metrics

# Configure logging.
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Backtest option pricing models over a historical period."
    )
    parser.add_argument(
        "--stock", type=str, required=True, help="Stock symbol (e.g., SPY)"
    )
    parser.add_argument(
        "--start-date", type=str, required=True, help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date", type=str, required=True, help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--option-type",
        type=str,
        choices=["C", "P"],
        default="C",
        help="Option type: C for call, P for put",
    )
    parser.add_argument(
        "--polygon-api-key", type=str, required=True, help="Polygon API key"
    )
    parser.add_argument("--fred-api-key", type=str, required=True, help="FRED API key")
    parser.add_argument(
        "--output",
        type=str,
        default="performance_metrics.csv",
        help="CSV file to store performance metrics",
    )
    args = parser.parse_args()

    stock_symbol = args.stock
    start_date = args.start_date
    end_date = args.end_date
    option_type = args.option_type
    polygon_api_key = args.polygon_api_key
    fred_api_key = args.fred_api_key
    output_file = args.output

    downloader = DataDownloader(polygon_api_key, fred_api_key)

    current_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")

    # Prepare output CSV for performance metrics.
    with open(output_file, mode="w", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["Date", "MAE", "MSE", "PricingTime_sec"])

    while current_date <= end_date_dt:
        date_str = current_date.strftime("%Y-%m-%d")
        logger.info("Processing date: %s", date_str)

        # Download stock data for the current date.
        stock_prices = downloader.download_stock_data(stock_symbol, date_str, date_str)
        if len(stock_prices) == 0:
            logger.warning("No stock data for %s; skipping.", date_str)
            current_date += timedelta(days=1)
            continue
        spot_price = stock_prices[-1]

        # Download option chain for the current date.
        option_chain = downloader.download_option_chain(
            stock_symbol, date_str, option_type=option_type
        )

        # For this backtest, assume options expire 30 days from the current date.
        maturity = 30 / 365.0

        # Download the risk‑free rate for the given maturity.
        rf_rate = downloader.download_risk_free_rate(maturity)

        # Create a Stock instance.
        volatility = (
            np.std(stock_prices) / np.mean(stock_prices)
            if np.mean(stock_prices) > 0
            else 0.2
        )
        dividend = 0.0  # Assume 0 dividend for simplicity.
        stock_instance = Stock(
            spot=spot_price,
            volatility=volatility,
            dividend=dividend,
            symbol=stock_symbol,
        )

        # Create a MarketEnvironment instance.
        market_env = MarketEnvironment(rate=rf_rate)

        # -------------------------------
        # Timing the pricing step.
        start_time = time.time()

        # Price the options using the Black‑Scholes vectorized function.
        priced_options = price_options_for_stock(
            stock_instance,
            market_env,
            option_chain,
            maturity,
            black_scholes_call_vectorized,
        )

        pricing_time = time.time() - start_time
        logger.info("Pricing completed in %.4f seconds on %s", pricing_time, date_str)
        # -------------------------------

        # Evaluate pricing performance.
        metrics = evaluate_performance(priced_options)
        logger.info(
            "Performance on %s: MAE=%.4f, MSE=%.4f",
            date_str,
            metrics["MAE"],
            metrics["MSE"],
        )

        # Append metrics and timing info to the output CSV.
        with open(output_file, mode="a", newline="") as f_out:
            writer = csv.writer(f_out)
            writer.writerow([date_str, metrics["MAE"], metrics["MSE"], pricing_time])

        current_date += timedelta(days=1)

    performance_csv = "performance_metrics.csv"
    plot_performance_metrics(performance_csv)

    logger.info("Backtesting complete. Performance metrics saved to %s", output_file)


if __name__ == "__main__":
    main()
