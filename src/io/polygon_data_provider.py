"""
polygon_data_provider.py

Implements a PolygonDataProvider class that fetches spot (or last-trade) prices
for symbols using the Polygon.io API. This is an example of integrating
a real external data source with the DataProvider interface.
"""

import abc
import logging
import requests
import threading
from typing import Optional, Dict, Any

from src.market.market_environment import DataProvider


class PolygonDataProvider(DataProvider):
    """
    A production-grade DataProvider implementation that fetches market data
    from the Polygon.io REST API.

    Polygon does NOT provide direct interest rates or implied volatilities
    at the free tier, so those methods raise NotImplementedError by default.
    """

    def __init__(
        self,
        api_key: str,
        session: Optional[requests.Session] = None,
        base_url: str = "https://api.polygon.io",
        timeout: float = 5.0,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize PolygonDataProvider with an API key and optional HTTP session.

        Args:
            api_key (str): Your Polygon.io API key.
            session (Optional[requests.Session]): Reusable HTTP session for connection pooling.
            base_url (str): Polygon.io base endpoint.
            timeout (float): Timeout in seconds for each API request.
            logger (Optional[logging.Logger]): Optional logger instance.
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.logger = logger if logger else logging.getLogger(__name__)

        # If no session provided, create a requests.Session for performance
        self.session = session or requests.Session()

        # Simple thread lock if we do multi-threaded calls
        self._lock = threading.Lock()

    def get_current_price(self, symbol: str) -> float:
        """
        Retrieve the current (last) price for the given symbol from Polygon.

        Args:
            symbol (str): The underlying symbol, e.g. "AAPL" or "SPY".

        Returns:
            float: The most recent traded price of the symbol.

        Raises:
            ValueError: If the data retrieved from Polygon is invalid or incomplete.
            requests.RequestException: If there's a network or HTTP error.
        """
        endpoint = f"{self.base_url}/v2/last/trade/{symbol.upper()}"
        params = {"apiKey": self.api_key}

        with self._lock:
            try:
                resp = self.session.get(endpoint, params=params, timeout=self.timeout)
                resp.raise_for_status()
            except requests.RequestException as e:
                self.logger.error("Polygon API request failed: %s", e)
                raise

            data = resp.json()
            # Example response structure (subject to Polygon's actual format):
            # {
            #   "status": "success",
            #   "symbol": "AAPL",
            #   "last": {
            #       "price": 150.23,
            #       "size": 100,
            #       "exchange": 11,
            #       "cond1": 14,
            #       ...
            #   }
            # }
            if "last" not in data or "price" not in data["last"]:
                self.logger.error("Invalid response from Polygon: %s", data)
                raise ValueError(f"Missing 'last.price' in response: {data}")

            price = float(data["last"]["price"])
            return price

    def get_interest_rate(self, key: str, **kwargs) -> float:
        """
        Polygon does not provide direct interest rates at the free tier.
        Raise NotImplementedError or implement an alternate data retrieval method.
        """
        raise NotImplementedError(
            "PolygonDataProvider.get_interest_rate() is not implemented; "
            "Polygon does not offer rates at this tier."
        )

    def get_volatility(
        self, symbol: str, maturity: float, strike: float, **kwargs
    ) -> float:
        """
        Polygon does not provide implied volatilities at the free tier.
        Raise NotImplementedError or implement a fallback approach.

        Args:
            symbol (str): Underlying symbol.
            maturity (float): Time to maturity in years.
            strike (float): Strike price.
            **kwargs: Additional parameters for your logic.
        """
        raise NotImplementedError(
            "PolygonDataProvider.get_volatility() is not implemented; "
            "Polygon does not offer implied vols at this tier."
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(base_url={self.base_url!r}, "
            f"api_key='***', timeout={self.timeout!r})"
        )
