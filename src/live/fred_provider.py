"""
fred_provider.py
================
This file defines a FRED data provider that extracts the risk–free rate from FRED.
It reads the API key and series ID from configuration (or kwargs) and saves the response
to artifacts. This is a real implementation that uses requests to call FRED’s API.
"""

import os
import datetime
import requests
import math
import yaml
from .base_data_provider import BaseDataProvider

class FredDataProvider(BaseDataProvider):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Load API key from kwargs or secrets configuration.
        self.api_key = kwargs.get("fred_api_key", self.secrets.get("fred", {}).get("api_key"))
        if not self.api_key:
            raise ValueError("FRED API key is missing.")
        # Load series_id from kwargs or secrets configuration.
        self.series_id = kwargs.get("fred_series_id", self.secrets.get("fred", {}).get("series_id", "DGS3MO"))
        # Load rate limit setting from kwargs or provider settings.
        self.rate_limit = kwargs.get("fred_rate_limit", self.provider_settings.get("fred", {}).get("rate_limit", 5))
    
    def getRiskFreeRate(self) -> float:
        """
        Retrieves the latest risk–free rate from FRED using its API.
        The function queries the observations endpoint and returns the most recent
        observation (converted to a decimal value).
        The result is saved (with timestamp) to the artifacts folder.
        """
        base_url = "https://api.stlouisfed.org/fred/series/observations"
        # For a robust call, fetch data from the last 30 days.
        start_date = (datetime.datetime.today() - datetime.timedelta(days=30)).strftime("%Y-%m-%d")
        params = {
            "series_id": self.series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "sort_order": "desc",
            "observation_start": start_date
        }
        
        response = requests.get(base_url, params=params)
        if response.status_code != 200:
            raise ValueError(f"FRED API request failed with status code {response.status_code}")
        
        data = response.json()
        observations = data.get("observations", [])
        if not observations:
            raise ValueError("No risk–free rate data returned from FRED.")
        
        # Assume the first observation is the most recent.
        rate_str = observations[0].get("value", None)
        if rate_str is None or rate_str == ".":
            raise ValueError("Risk–free rate value is missing in FRED API response.")
        
        # FRED yields are typically in percent; convert to a decimal.
        rate = float(rate_str) / 100.0
        
        # Save artifact.
        today_str = datetime.datetime.now().strftime("%Y%m%d")
        artifact_data = {"series_id": self.series_id, "risk_free_rate": rate}
        self.save_artifact("artifacts/fred_rates", f"fred_rates_{today_str}.csv", artifact_data)
        
        return rate

if __name__ == "__main__":
    # For testing: Ensure you have a valid FRED API key in config/secrets.yaml.
    provider = FredDataProvider(fred_api_key="YOUR_FRED_API_KEY")
    rate = provider.getRiskFreeRate()
    print("Risk-free rate from FRED =", rate)
