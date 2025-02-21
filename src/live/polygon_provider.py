"""
polygon_provider.py
===================
This file defines a Polygon data provider that retrieves market data for a given stock symbol.
It calls Polygon's ticker snapshot API and returns a dictionary with keys:
  - 'spot': the current stock price.
  - 'volatility': the implied volatility (if available; defaults to 0.2 otherwise).
  - Optionally, 'dividend', 'discrete_dividend', and 'dividend_times'.
If essential data is missing, an error is raised.
The raw response is saved (with a timestamp) to artifacts/polygon_data/.
"""

import os
import datetime
import requests
import yaml
from .base_data_provider import BaseDataProvider

class PolygonDataProvider(BaseDataProvider):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Load API key from kwargs or secrets configuration.
        self.api_key = kwargs.get("polygon_api_key", self.secrets.get("polygon", {}).get("api_key"))
        if not self.api_key:
            raise ValueError("Polygon API key is missing.")
        # Load rate limit from kwargs or provider settings.
        self.rate_limit = kwargs.get("polygon_rate_limit", self.provider_settings.get("polygon", {}).get("rate_limit", 10))
    
    def getMarketData(self, symbol: str) -> dict:
        """
        Retrieves market data for the given symbol from Polygon.
        This implementation uses the ticker snapshot endpoint.
        
        Expected data includes:
            - 'spot': current stock price (from the "last" field, or fallback to "prevClose").
            - 'volatility': implied volatility (if provided; defaults to 0.2 otherwise).
            - Optionally: 'dividend', 'discrete_dividend', 'dividend_times'.
        
        Raises an error if essential data is missing.
        Saves the raw data artifact with a timestamp to artifacts/polygon_data/.
        """
        base_url = f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}"
        params = {"apiKey": self.api_key}
        
        response = requests.get(base_url, params=params)
        if response.status_code != 200:
            raise ValueError(f"Polygon API request failed with status code {response.status_code}")
        
        data = response.json()
        if data.get("status") != "OK":
            raise ValueError(f"Polygon API error for symbol {symbol}: {data}")
        
        ticker = data.get("ticker", {})
        if not ticker:
            raise ValueError(f"No ticker data returned for symbol {symbol}")
        
        # Extract the spot price. Prefer "last" price; if missing, try "prevClose".
        spot = ticker.get("last")
        if spot is None:
            spot = ticker.get("prevClose")
        if spot is None:
            raise ValueError(f"Spot price not available for symbol {symbol}")
        
        # Extract volatility if available; otherwise default to 0.2.
        # (Polygon may not provide implied volatility directly for equities.)
        volatility = ticker.get("day", {}).get("volatility", 0.2)
        if volatility is None:
            volatility = 0.2
        
        # Optionally extract dividend info.
        dividend = ticker.get("dividend", 0.0)
        discrete_dividend = ticker.get("discrete_dividend", None)
        dividend_times = ticker.get("dividend_times", None)
        
        market_data = {
            "spot": float(spot),
            "volatility": float(volatility),
            "dividend": float(dividend)
        }
        if discrete_dividend is not None:
            market_data["discrete_dividend"] = discrete_dividend
        if dividend_times is not None:
            market_data["dividend_times"] = dividend_times
        
        # Save artifact to artifacts/polygon_data.
        today_str = datetime.datetime.now().strftime("%Y%m%d")
        artifact_data = {"symbol": symbol, "market_data": market_data}
        self.save_artifact("artifacts/polygon_data", f"polygon_data_{today_str}.csv", artifact_data)
        
        return market_data

if __name__ == "__main__":
    # For testing, replace 'AAPL' with a valid symbol.
    provider = PolygonDataProvider(polygon_api_key="YOUR_POLYGON_API_KEY")
    md = provider.getMarketData("AAPL")
    print("Market data for AAPL:", md)
