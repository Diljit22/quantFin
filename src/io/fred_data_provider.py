"""
fred_data_provider.py

Implements a FredDataProvider class that fetches interest rate data from
the Federal Reserve's FRED API. This is a production-grade example for
fetching free interest rate data, which you can integrate into your
MarketEnvironment in place of (or in addition to) local or constant rates.

References:
    FRED API docs: https://fred.stlouisfed.org/docs/api/fred/

Example usage:
    dp = FredDataProvider(api_key="YOUR_FRED_API_KEY")
    rate = dp.get_interest_rate("DGS10")  # e.g., 10-year Treasury yield
    print("Latest 10-year yield:", rate)
"""

import logging
import requests
import threading
from typing import Optional, Any

from src.market.market_environment import DataProvider


class FredDataProvider(DataProvider):
    """
    A production-grade DataProvider that fetches (mostly) daily interest rates
    from the Federal Reserve's FRED API. Typically used for U.S. Treasury yields,
    but can fetch any FRED series.

    FRED's free API often provides EOD (end-of-day) or daily values,
    so it's not truly real-time.
    """

    def __init__(
        self,
        api_key: str,
        session: Optional[requests.Session] = None,
        logger: Optional[logging.Logger] = None,
        base_url: str = "https://api.stlouisfed.org",
        timeout: float = 5.0,
    ):
        """
        Initialize FredDataProvider with a FRED API key and optional HTTP session.

        Args:
            api_key (str): Your FRED API key from the Federal Reserve website.
            session (Optional[requests.Session]): Reusable HTTP session for better performance.
            logger (Optional[logging.Logger]): Optional logger instance for error/info messages.
            base_url (str): Base FRED endpoint (rarely changed).
            timeout (float): Timeout in seconds for each API request.
        """
        self.api_key = api_key
        self.session = session or requests.Session()
        self.logger = logger if logger else logging.getLogger(__name__)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        self._lock = threading.Lock()  # for thread safety

    def get_current_price(self, symbol: str) -> float:
        """
        Not implemented for interest-rate–only data. FRED doesn't provide
        real-time 'spot prices' in the equity sense. Raise NotImplementedError.
        """
        raise NotImplementedError(
            "FredDataProvider does not fetch equity spot prices. "
            "Use PolygonDataProvider or another provider for that."
        )

    def get_interest_rate(self, key: str, **kwargs) -> float:
        """
        Fetch the most recent interest rate (annualized, in decimal form)
        from FRED for the given 'key' (e.g., 'DGS10' for 10-year Treasury).

        By default, returns the latest available observation. The rate is typically
        expressed as a percentage (e.g., '3.50'), so we convert it to decimal (0.0350).

        Args:
            key (str): FRED series ID. For instance:
                - 'DGS1' for 1-year Treasury
                - 'DGS10' for 10-year Treasury
                - 'DFF' for Fed Funds rate
                - ... (check FRED for more series)
            **kwargs: Additional parameters (e.g., 'observation_start', 'observation_end')
                if you want to fetch a specific date range. See FRED API docs.

        Returns:
            float: The annualized rate in decimal form (e.g., 0.035 for 3.5%).

        Raises:
            ValueError: If the API response is invalid or no observations are found.
            requests.RequestException: If there's a network or HTTP error.
        """
        endpoint = f"{self.base_url}/fred/series/observations"
        params = {
            "api_key": self.api_key,
            "series_id": key,
            "file_type": "json",
        }
        # Merge any additional kwargs (e.g. 'observation_start')
        params.update(kwargs)

        with self._lock:
            try:
                resp = self.session.get(endpoint, params=params, timeout=self.timeout)
                resp.raise_for_status()
            except requests.RequestException as e:
                self.logger.error("FRED API request failed: %s", e)
                raise

            data = resp.json()
            # Expect a structure like:
            # {
            #   "realtime_start":"2023-01-01",
            #   "realtime_end":"2023-01-01",
            #   "observation_start":"1776-07-04",
            #   "observation_end":"9999-12-31",
            #   "units":"lin",
            #   "output_type":1,
            #   "file_type":"json",
            #   "order_by":"observation_date",
            #   "sort_order":"asc",
            #   "count":XXX,
            #   "offset":0,
            #   "limit":100000,
            #   "observations":[
            #       {"realtime_start":"2023-08-10","realtime_end":"2023-08-10","date":"2023-08-09","value":"4.02"},
            #       ...
            #    ]
            # }

            if "observations" not in data or len(data["observations"]) == 0:
                raise ValueError(
                    f"No observations found for series {key} in response: {data}"
                )

            # last observation is the one we want:
            last_obs = data["observations"][-1]
            val_str = last_obs.get("value", None)
            if val_str in (None, ".", ""):
                # Some series use '.' to indicate no data for that date
                raise ValueError(
                    f"Invalid or missing rate value in last observation: {last_obs}"
                )
            try:
                # yields given in percent (e.g., "4.02" = 4.02%)
                rate_decimal = float(val_str) / 100.0
            except ValueError:
                raise ValueError(f"Cannot convert '{val_str}' to float in {last_obs}")

            return rate_decimal

    def get_volatility(
        self, symbol: str, maturity: float, strike: float, **kwargs
    ) -> float:
        """
        Not implemented. FRED typically does not provide implied volatilities.
        """
        raise NotImplementedError(
            "FredDataProvider.get_volatility() not supported. FRED does not offer implied vols."
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(base_url={self.base_url!r}, "
            f"api_key='***', timeout={self.timeout!r})"
        )
