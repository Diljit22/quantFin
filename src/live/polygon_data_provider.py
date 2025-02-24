#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
polygon_data_provider.py
========================
Provides a unified interface for fetching market data (stocks and options) from
the Polygon API.

Features
--------
- Parse OSI-formatted option symbols to extract underlying, expiration, type, strike.
- Fetch spot price from Polygon (with fallback to previous close).
- Fetch historical daily closing prices and compute annualized historical volatility.
- Optionally retrieve additional data such as dividends or discrete_dividends
    (not fully implemented).
- Mini-CLI usage for either a stock symbol or an OSI-formatted option symbol.

Usage
-----
    python polygon_data_provider.py --mode stock --symbol AAPL
    python polygon_data_provider.py --mode option --option-symbol O:SPY210917C00450000
    provider = PolygonDataProvider()
    provider.main_cli()

Configuration
-------------
Expects to find:
    configs/secrets.yaml:
        polygon:
            api_key: <YOUR_POLYGON_API_KEY>
    configs/provider_settings.yaml:
        polygon:
            rate_limit: 10
These can be overridden via constructor kwargs (polygon_api_key, polygon_rate_limit).
"""

import sys
import math
import logging
import requests
import statistics
import re
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

# If YAML is available, load config. Otherwise, skip silently or handle error.
try:
    import yaml
except ImportError:
    yaml = None

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


class PolygonDataProvider:
    """
    Provides methods to fetch market data (spot prices, historical prices, etc.)
    for equities and to parse/compute data for options from the Polygon API.

    Parameters
    ----------
    polygon_api_key : str, optional
        Override for the Polygon API key (default None).
        If not provided, it tries to load from secrets config.
    polygon_rate_limit : int, optional
        Override for the rate limit (default None).
        If not provided, it tries to load from provider settings.
    secrets_path : str, optional
        Path to secrets.yaml. Defaults to "configs/secrets.yaml".
    provider_settings_path : str, optional
        Path to provider_settings.yaml. Defaults to "configs/provider_settings.yaml".

    Notes
    -----
    This class does not implement actual rate limiting (sleep, token bucket, etc.).
    The rate limit setting is available if you need to add throttling logic.
    """

    BASE_URL = "https://api.polygon.io"

    def __init__(
        self,
        polygon_api_key: Optional[str] = None,
        polygon_rate_limit: Optional[int] = None,
        secrets_path: str = "configs/secrets.yaml",
        provider_settings_path: str = "configs/provider_settings.yaml",
        **kwargs,
    ):
        self.secrets = self._load_yaml(secrets_path) if yaml else {}
        self.provider_settings = self._load_yaml(provider_settings_path) if yaml else {}

        # API key from constructor or configs.
        self.api_key = polygon_api_key or self.secrets.get("polygon", {}).get("api_key")
        if not self.api_key:
            raise ValueError(
                "Polygon API key is missing in configs or constructor arguments."
            )

        # Rate limit from constructor or configs.
        self.rate_limit = polygon_rate_limit or self.provider_settings.get(
            "polygon", {}
        ).get("rate_limit", 10)

    def _load_yaml(self, path: str) -> dict:
        """
        Load a YAML file from the specified path.

        Parameters
        ----------
        path : str
            File path to the YAML config.

        Returns
        -------
        dict
            Parsed YAML content, or empty dict on error.
        """
        try:
            with open(path, "r") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            logging.warning("Config file not found: %s", path)
            return {}
        except Exception as e:
            logging.error("Error loading YAML from %s: %s", path, e)
            return {}

    def get_spot_price(self, stock_symbol: str) -> float:
        """
        Fetch the latest spot price for a given stock symbol from Polygon,
        falling back to previous close if realtime data is forbidden.

        Parameters
        ----------
        stock_symbol : str
            Stock ticker symbol (e.g., 'AAPL').

        Returns
        -------
        float
            Latest spot price or previous close price.

        Raises
        ------
        SystemExit
            If the data is invalid or calls fail.
        """
        # Attempt to fetch realtime last trade
        url = f"{self.BASE_URL}/v2/last/trade/{stock_symbol}"
        params = {"apiKey": self.api_key}
        response = None
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if "last" not in data or "price" not in data["last"]:
                logging.error("Invalid response structure for spot price: %s", data)
                sys.exit(1)
            price = data["last"]["price"]
            logging.info("Fetched realtime spot price for %s: %s", stock_symbol, price)
            return float(price)
        except requests.RequestException:
            if response is not None and response.status_code == 403:
                logging.warning(
                    "Realtime access forbidden. Falling back to previous close."
                )
                return self.get_previous_close(stock_symbol)
            else:
                logging.exception("Error fetching spot price for %s", stock_symbol)
                sys.exit(1)

    def get_previous_close(self, stock_symbol: str) -> float:
        """
        Fetch the previous close price for a given stock symbol.

        Parameters
        ----------
        stock_symbol : str
            The stock ticker symbol.

        Returns
        -------
        float
            The previous close price.

        Raises
        ------
        SystemExit
            If the API call fails or invalid data is returned.
        """
        url = f"{self.BASE_URL}/v2/aggs/ticker/{stock_symbol}/prev"
        params = {"apiKey": self.api_key, "adjusted": "true"}
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if "results" not in data or not data["results"]:
                logging.error("Invalid previous close data: %s", data)
                sys.exit(1)
            price = data["results"][0]["c"]
            logging.info("Fetched previous close for %s: %s", stock_symbol, price)
            return float(price)
        except requests.RequestException as e:
            logging.error("Error fetching previous close for %s: %s", stock_symbol, e)
            sys.exit(1)

    def get_historical_prices(
        self, stock_symbol: str, from_date: str, to_date: str
    ) -> list:
        """
        Fetch historical daily closing prices for the stock between two dates.

        Parameters
        ----------
        stock_symbol : str
            The stock ticker symbol (e.g., 'AAPL').
        from_date : str
            Start date in 'YYYY-MM-DD' format.
        to_date : str
            End date in 'YYYY-MM-DD' format.

        Returns
        -------
        list
            List of daily closing prices (floats).

        Raises
        ------
        SystemExit
            If the request fails or no valid data is returned.
        """
        url = f"{self.BASE_URL}/v2/aggs/ticker/{stock_symbol}/range/1/day/{from_date}/{to_date}"
        params = {"apiKey": self.api_key, "adjusted": "true"}
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if "results" not in data:
                logging.error("Invalid historical data response: %s", data)
                sys.exit(1)
            prices = [float(res["c"]) for res in data["results"] if "c" in res]
            if not prices:
                logging.error("No historical prices found for %s", stock_symbol)
                sys.exit(1)
            logging.info(
                "Fetched %d historical prices for %s", len(prices), stock_symbol
            )
            return prices
        except requests.RequestException as e:
            logging.error(
                "Error fetching historical prices for %s: %s", stock_symbol, e
            )
            sys.exit(1)

    def compute_historical_volatility(self, prices: list) -> float:
        """
        Compute annualized historical volatility (as %) from daily closing prices.

        Uses the standard deviation of log returns multiplied by sqrt(252).

        Parameters
        ----------
        prices : list
            A list of daily closing prices.

        Returns
        -------
        float
            Annualized volatility as a percentage (e.g., 30.0 for 30%).

        Raises
        ------
        SystemExit
            If insufficient price data is provided.
        """
        if len(prices) < 2:
            logging.error("Not enough price data to compute volatility.")
            sys.exit(1)
        log_returns = []
        for i in range(1, len(prices)):
            prev_price = prices[i - 1]
            curr_price = prices[i]
            try:
                log_return = math.log(curr_price / prev_price)
                log_returns.append(log_return)
            except ValueError:
                logging.error(
                    "Encountered non-positive price when computing log return."
                )
                sys.exit(1)
            except Exception as e:
                logging.error("Error computing log return: %s", e)
                sys.exit(1)

        stdev = statistics.stdev(log_returns)
        annualized_vol = stdev * math.sqrt(252)
        return annualized_vol * 100.0  # convert to percentage

    def parse_option_symbol(self, option_symbol: str) -> Dict[str, Any]:
        """
        Parse OSI option symbol; extract underlying, expiration, type, strike.

        Parameters
        ----------
        option_symbol : str
            The option symbol (e.g. 'O:SPY210917C00450000').

        Returns
        -------
        dict
            Fields: underlying (str), expiration (datetime),
                    option_type (str), strike (float).

        Raises
        ------
        SystemExit
            If the symbol does not match the OSI format or parsing fails.
        """
        # Strip prefix like 'O:' if present
        if option_symbol.startswith("O:"):
            option_symbol = option_symbol[2:]

        match = re.match(r"([A-Z]+)(\d{6})([CP])(\d{8})", option_symbol)
        if not match:
            logging.error("Option symbol not in OSI format: %s", option_symbol)
            sys.exit(1)

        underlying = match.group(1)
        exp_str = match.group(2)
        option_type = match.group(3)
        strike_str = match.group(4)

        # Expiration
        try:
            # parse YYMMDD => e.g. "210917"
            year = int(exp_str[0:2])
            year += 2000 if year < 50 else 1900
            month = int(exp_str[2:4])
            day = int(exp_str[4:6])
            expiration = datetime(year, month, day)
        except Exception as e:
            logging.error("Failed to parse expiration date from symbol: %s", e)
            sys.exit(1)

        # Strike
        try:
            strike = float(strike_str) / 1000.0
        except Exception as e:
            logging.error("Failed to parse strike price from symbol: %s", e)
            sys.exit(1)

        return {
            "underlying": underlying,
            "expiration": expiration,
            "option_type": option_type,
            "strike": strike,
        }

    def get_option_data(self, option_symbol: str) -> Dict[str, Any]:
        """
        Fetch relevant data for a given option, including:
          - parsed OSI fields (underlying, expiration, type, strike),
          - spot price,
          - historical volatility from ~60 trading days,
          - placeholders for dividend info (not implemented).

        Parameters
        ----------
        option_symbol : str
            OSI-formatted symbol (e.g. 'O:SPY210917C00450000').

        Returns
        -------
        dict
            Contains 'underlying', 'expiration', 'option_type', 'strike', 'spot_price',
            'historical_volatility', and placeholders for 'dividend_info'.

        Raises
        ------
        SystemExit
            If any step in the process fails.
        """
        # Parse OSI
        option_data = self.parse_option_symbol(option_symbol)
        underlying = option_data["underlying"]

        # Spot price
        spot_price = self.get_spot_price(underlying)

        # Historical prices -> historical volatility
        to_date = datetime.now().date()
        from_date = to_date - timedelta(days=90)  # ~60 trading days
        hist_prices = self.get_historical_prices(
            underlying,
            from_date.strftime("%Y-%m-%d"),
            to_date.strftime("%Y-%m-%d"),
        )
        hist_vol = self.compute_historical_volatility(hist_prices)

        # Placeholder for dividend info
        dividend_info = "Not Implemented"

        return {
            "underlying": underlying,
            "expiration": option_data["expiration"],
            "option_type": option_data["option_type"],
            "strike": option_data["strike"],
            "spot_price": spot_price,
            "historical_volatility": hist_vol,
            "dividend_info": dividend_info,
        }

    def main_cli(self):
        """
        Mini-CLI entry point to fetch data for a stock symbol or an option symbol.

        Usage
        -----
            python polygon_data_provider.py --mode stock --symbol AAPL
            python polygon_data_provider.py --mode option --option-symbol O:SPY210917C00450000
        """
        import argparse

        parser = argparse.ArgumentParser(description="Fetch market data via Polygon.")
        parser.add_argument(
            "--mode", choices=["stock", "option"], default="stock", help="Data mode"
        )
        parser.add_argument(
            "--symbol",
            type=str,
            default=None,
            help="Stock symbol (e.g., 'AAPL') when mode=stock",
        )
        parser.add_argument(
            "--option-symbol",
            type=str,
            default=None,
            help="Option symbol in OSI format when mode=option",
        )

        args = parser.parse_args()

        if args.mode == "stock":
            if not args.symbol:
                logging.error("Please provide --symbol for stock mode.")
                sys.exit(1)
            logging.info("Fetching stock data for %s", args.symbol)
            spot = self.get_spot_price(args.symbol)
            logging.info("Spot price for %s: %.4f", args.symbol, spot)
            # Optionally fetch historical prices or do more...
        else:  # option
            if not args.option_symbol:
                logging.error("Please provide --option-symbol for option mode.")
                sys.exit(1)
            logging.info("Fetching option data for %s", args.option_symbol)
            data = self.get_option_data(args.option_symbol)
            logging.info("Option Data: %s", data)
            print("Option Data Extraction:")
            print(f"  Underlying: {data['underlying']}")
            print(f"  Spot Price: {data['spot_price']:.4f}")
            print(f"  Hist Vol: {data['historical_volatility']:.2f}%")
            print(f"  Strike: {data['strike']}")
            print(f"  Expiration: {data['expiration'].strftime('%Y-%m-%d')}")
            print(f"  Dividend Info: {data['dividend_info']}")
