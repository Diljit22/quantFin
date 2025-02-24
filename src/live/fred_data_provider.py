#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fred_data_provider.py
=====================
Consolidates logic for fetching various Treasury yields and rates from the FRED API.

This module reads FRED API settings from configurations or
passed arguments. It features a FredDataProvider class with methods to fetch:
  - Short-term rates (e.g., 3-month T-bills),
  - Long-term yields (interpolated or direct),
  - Federal Funds rates,
  - Generic series observations.

Usage
-----
To instantiate FredDataProvider and call the relevant methods:

    from live import FredDataProvider

    provider = FredDataProvider()
    provider.main_cli()
    short_rate = provider.get_short_term_rate(series_id="DTB3")
    long_rate = provider.get_long_term_rate(maturity=5.0)

Configuration
-------------
API key and rate limit can be stored in:
    configs/secrets.yaml
    configs/provider_settings.yaml

Override default configs via kwargs (e.g., `FredDataProvider(fred_api_key="...")`).
"""

import sys
import logging
import requests
from typing import Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


class FredDataProvider:
    """
    Provides methods to fetch Treasury rates and other series from the FRED API.

    Parameters
    ----------
    fred_api_key : str, optional
        FRED API key override. If not provided, attempts to load from secrets config.
    fred_rate_limit : int, optional
        Rate limit for FRED API calls. If not provided, attempts to load
        from provider settings.
    secrets_path : str, optional
        Path to secrets.yaml file containing 'fred.api_key'.
        Default is 'configs/secrets.yaml'.
    provider_settings_path : str, optional
        Path to provider_settings.yaml containing 'fred.rate_limit'.
        Default is 'configs/provider_settings.yaml'.
    """

    BASE_URL = "https://api.stlouisfed.org/fred"

    def __init__(
        self,
        fred_api_key: Optional[str] = None,
        fred_rate_limit: Optional[int] = None,
        secrets_path: str = "configs/secrets.yaml",
        provider_settings_path: str = "configs/provider_settings.yaml",
        **kwargs,
    ):
        # Load secrets
        self.secrets = self._load_yaml(secrets_path) if yaml else {}
        # Load provider settings
        self.provider_settings = self._load_yaml(provider_settings_path) if yaml else {}

        # Set API Key
        self.api_key = fred_api_key or self.secrets.get("fred", {}).get("api_key")
        if not self.api_key:
            raise ValueError("FRED API key not found in configs or arguments.")

        # Set rate limit
        self.rate_limit = fred_rate_limit or self.provider_settings.get("fred", {}).get(
            "rate_limit", 5
        )

    def _load_yaml(self, path: str) -> dict:
        """
        Load a YAML file from the specified path.

        Parameters
        ----------
        path : str
            Filesystem path to the YAML file.

        Returns
        -------
        dict
            The parsed YAML content as a Python dictionary.
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

    def fetch_series_observations(self, series_id: str, **kwargs) -> dict:
        """
        Fetch observations for a given FRED series.

        Parameters
        ----------
        series_id : str
            The FRED series ID (e.g., 'DTB3' for 3-Month Treasury Bill).
        **kwargs : dict
            Additional query parameters for the API request.

        Returns
        -------
        dict
            JSON response as a dictionary (parsed from FRED's API).

        Raises
        ------
        SystemExit
            If there's an error fetching data from the API.
        """
        url = f"{self.BASE_URL}/series/observations"
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
        }
        params.update(kwargs)

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            logging.debug("FRED API response (%s): %s", series_id, response.text)
            return response.json()
        except requests.RequestException as e:
            logging.error(
                "Error fetching data from FRED API for series %s: %s", series_id, e
            )
            sys.exit(1)

    def get_short_term_rate(self, series_id: str = "DTB3") -> float:
        """
        Retrieve the latest short-term Treasury Bill rate from FRED
        e.g., 3-month T-bill.

        Parameters
        ----------
        series_id : str
            FRED series ID for the short-term T-bill rate (default is 'DTB3').

        Returns
        -------
        float
            Latest T-bill rate (as a decimal, e.g., 0.02 for 2%).

        Raises
        ------
        ValueError
            If no observations are found or the rate is invalid.
        """
        data = self.fetch_series_observations(series_id)
        observations = data.get("observations", [])
        if not observations:
            raise ValueError(f"No observations returned for series: {series_id}")

        # Find the latest observation by date.
        latest = max(observations, key=lambda x: x["date"])
        rate_str = latest.get("value", ".")
        try:
            rate = float(rate_str) if rate_str != "." else None
        except ValueError:
            raise ValueError(f"Invalid rate value received: {rate_str}")

        if rate is None:
            raise ValueError("No valid short-term rate data available.")

        # FRED yields are typically in percent; convert to decimal.
        return rate / 100.0

    def get_federal_funds_rate(self, series_id: str = "FEDFUNDS") -> float:
        """
        Retrieve the latest Federal Funds Rate (overnight lending rate) from FRED.

        Parameters
        ----------
        series_id : str
            FRED series ID for the Fed Funds rate (default is 'FEDFUNDS').

        Returns
        -------
        float
            Latest Fed Funds rate (as a decimal).

        Raises
        ------
        ValueError
            If no observations are found or the rate is invalid.
        """
        data = self.fetch_series_observations(series_id)
        observations = data.get("observations", [])
        if not observations:
            raise ValueError(f"No observations returned for series: {series_id}")
        latest = max(observations, key=lambda x: x["date"])
        rate_str = latest.get("value", ".")
        try:
            rate = float(rate_str) if rate_str != "." else None
        except ValueError:
            raise ValueError(f"Invalid Fed Funds rate value: {rate_str}")

        if rate is None:
            raise ValueError("No valid Fed Funds rate data available.")

        return rate / 100.0

    def get_generic_rate(self, series_id: str) -> float:
        """
        Retrieve the latest rate (or yield) for a generic FRED series.

        Parameters
        ----------
        series_id : str
            The FRED series ID (e.g., 'DGS10' for 10-year treasury).

        Returns
        -------
        float
            Latest rate or yield (as a decimal).

        Raises
        ------
        ValueError
            If no observations are found or the rate is invalid.
        """
        data = self.fetch_series_observations(series_id)
        obs = data.get("observations", [])
        if not obs:
            raise ValueError(f"No observations found for series: {series_id}")
        latest = max(obs, key=lambda x: x["date"])
        rate_str = latest.get("value", ".")
        try:
            rate = float(rate_str) if rate_str != "." else None
        except ValueError:
            raise ValueError(f"Invalid rate value for series {series_id}: {rate_str}")

        if rate is None:
            raise ValueError(f"No valid data for series {series_id}.")

        return rate / 100.0

    def get_yield_curve(self, series_map: Dict[float, str]) -> Dict[float, float]:
        """
        Fetch the latest yields for multiple maturities from FRED concurrently.

        Parameters
        ----------
        series_map : dict
            A dictionary mapping maturity (float) to FRED series ID (str).

        Returns
        -------
        dict
            A dictionary mapping maturity (float) to yield (float, in decimal form).

        Raises
        ------
        SystemExit
            If an error occurs fetching data.
        """
        yield_curve = {}

        def _fetch_latest_rate(maturity: float, sid: str) -> float:
            rate = self.get_generic_rate(sid)
            logging.info(
                "Fetched yield for %s-year maturity from series %s: %.4f",
                maturity,
                sid,
                rate,
            )
            return rate

        with ThreadPoolExecutor(max_workers=len(series_map)) as executor:
            futures = {
                executor.submit(_fetch_latest_rate, m, s): m
                for m, s in series_map.items()
            }
            for fut in as_completed(futures):
                maturity = futures[fut]
                yield_curve[maturity] = fut.result()

        return yield_curve

    def interpolate_yield(
        self, target_maturity: float, yield_curve: Dict[float, float]
    ) -> float:
        """
        Interpolate a yield for the given target maturity from a yield curve.

        Parameters
        ----------
        target_maturity : float
            The maturity in years (e.g., 4.5).
        yield_curve : dict
            Maps maturity (float) to yield (float, as decimal).

        Returns
        -------
        float
            The interpolated yield (as decimal).

        Raises
        ------
        ValueError
            If interpolation fails.
        """
        maturities = sorted(yield_curve.keys())
        # If exact match
        if target_maturity in yield_curve:
            return yield_curve[target_maturity]

        # Extrapolate or find bracket
        if target_maturity < maturities[0]:
            lower, upper = maturities[0], maturities[1]
        elif target_maturity > maturities[-1]:
            lower, upper = maturities[-2], maturities[-1]
        else:
            lower, upper = None, None
            for i in range(len(maturities) - 1):
                if maturities[i] < target_maturity < maturities[i + 1]:
                    lower = maturities[i]
                    upper = maturities[i + 1]
                    break

        if lower is None or upper is None:
            raise ValueError("Unable to interpolate: no suitable bracket found.")

        rate_lower = yield_curve[lower]
        rate_upper = yield_curve[upper]
        # Linear interpolation
        slope = (rate_upper - rate_lower) / (upper - lower)
        return rate_lower + (target_maturity - lower) * slope

    def get_long_term_rate(self, maturity: float) -> float:
        """
        Retrieve the long-term rate for a given maturity by interpolating a yield curve.

        Parameters
        ----------
        maturity : float
            The target maturity in years (e.g., 1.34 or 5.0).

        Returns
        -------
        float
            The interpolated long-term rate (as a decimal).

        Raises
        ------
        ValueError
            If interpolation fails.
        """
        # Define a default series map for long-term yields.
        default_series_map = {
            1: "DGS1",
            2: "DGS2",
            3: "DGS3",
            5: "DGS5",
            7: "DGS7",
            10: "DGS10",
            20: "DGS20",
            30: "DGS30",
        }
        yield_curve = self.get_yield_curve(default_series_map)
        return self.interpolate_yield(maturity, yield_curve)

    def main_cli(self):
        """
        CLI usage: fetch and print short-term, FedFunds, generic, long-term rates.

        Usage:
            python fred_data_provider.py --mode short-term
            python fred_data_provider.py --mode fedfunds
            python fred_data_provider.py --mode generic --series_id DGS10
            python fred_data_provider.py --mode long-term --maturity 5.0
        """
        import argparse

        parser = argparse.ArgumentParser(description="FRED Data Provider CLI")
        parser.add_argument(
            "--mode",
            choices=["short-term", "fedfunds", "generic", "long-term"],
            default="short-term",
            help="Which rate to fetch.",
        )
        parser.add_argument(
            "--series_id",
            type=str,
            default="DTB3",
            help="FRED series ID to fetch if mode is 'generic'.",
        )
        parser.add_argument(
            "--maturity",
            type=float,
            default=5.0,
            help="Target maturity in years for long-term rate interpolation.",
        )
        args = parser.parse_args()

        if args.mode == "short-term":
            rate = self.get_short_term_rate()
            logging.info("Short-Term (3-mo T-bill) Rate: %.4f", rate)
        elif args.mode == "fedfunds":
            rate = self.get_federal_funds_rate()
            logging.info("Federal Funds Rate: %.4f", rate)
        elif args.mode == "generic":
            rate = self.get_generic_rate(args.series_id)
            logging.info("%s: %.4f", args.series_id, rate)
        elif args.mode == "long-term":
            rate = self.get_long_term_rate(args.maturity)
            logging.info(
                "Long-Term Rate (for %.2f-year maturity): %.4f", args.maturity, rate
            )
