from typing import Any


class DataProvider:
    """
    Abstract base class for external or real-time market data feeds.
    Implementations can fetch real-time quotes/historical data.
    """

    def get_current_price(self, symbol: str) -> float:
        """
        Retrieve the most recent available spot price for the given symbol.
        This method is meant to be overridden in a subclass.

        Parameters
        ----------
        symbol : str
            Symbol or ticker for which to retrieve the price.

        Returns
        -------
        float
            The current spot price.

        Raises
        ------
        NotImplementedError
            If not overridden by a subclass.
        """
        raise NotImplementedError("get_current_price not implemented.")

    def get_interest_rate(self, key: str, **kwargs) -> float:
        """
        Retrieve interest rate data keyed by a curve or tenor.

        Parameters
        ----------
        key : str
            Identifier for the rate (e.g., a particular tenor or curve name).
        **kwargs : dict
            Additional parameters for rate retrieval.

        Returns
        -------
        float
            The interest rate.

        Raises
        ------
        NotImplementedError
            If not overridden by a subclass.
        """
        raise NotImplementedError("Interest rate retrieval not implemented.")

    def get_volatility(
        self, symbol: str, maturity: float, strike: float, **kwargs
    ) -> float:
        """
        Retrieve implied volatility from an external feed for a given symbol,
        strike, and maturity.

        Parameters
        ----------
        symbol : str
            The underlying symbol.
        maturity : float
            Time to maturity in years.
        strike : float
            Strike price.
        **kwargs : dict
            Additional parameters (e.g., date, data source).

        Returns
        -------
        float
            The implied volatility as a decimal (e.g., 0.20 = 20%).

        Raises
        ------
        NotImplementedError
            If not overridden by a subclass.
        """
        raise NotImplementedError(
            "Volatility retrieval not implemented in DataProvider."
        )
