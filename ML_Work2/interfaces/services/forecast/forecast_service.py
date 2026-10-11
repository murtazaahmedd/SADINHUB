"""Interface for training, replaying and forecasting."""
from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, Tuple

import pandas as pd


class IForecastService(ABC):
    """
    #Feature: Forecasting
    #Scenario: Callers depend on this interface, never on the model classes
    #Complexity: Time O(months * fit) - one model per month

    Walk-forward replay, year-end forecast and long-range accuracy.
    """

    @abstractmethod
    def build_panel(self, daily: pd.DataFrame, weather: pd.DataFrame, end_date):
        """
        #Feature: Forecasting
        #Scenario: The meter x date arrays the model reads
        #Complexity: Time O(m*n) - meters times days

        Args:
            daily: The daily table.
            weather: Observed daily temperature.
            end_date: Last day the panel covers.

        Returns:
            A Panel.

        Raises:
            ValueError: When an argument is None.
        """

    @abstractmethod
    def walk_forward(self, panel, first_origin, last_origin, horizons: Optional[int] = None,
                     log: Optional[Callable[[str], None]] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        #Feature: Forecasting
        #Scenario: Replay every morning exactly as it would have run
        #Complexity: Time O(months * fit + days * rows) - monthly models plus daily features

        Args:
            panel: The panel.
            first_origin: First issue day.
            last_origin: Last issue day.
            horizons: Days ahead issued on each morning.
            log: Progress callback.

        Returns:
            (forecasts for every morning, learned coefficients of each monthly model).

        Raises:
            ValueError: When panel is None.
        """

    @abstractmethod
    def forecast_to_year_end(self, panel, daily: pd.DataFrame, forecast_end) -> Tuple[pd.DataFrame, object, pd.Timestamp]:
        """
        #Feature: Forecasting
        #Scenario: Train on every complete day and forecast each day to the year end
        #Complexity: Time O(fit + m*h) - one fit plus meters times horizons

        Args:
            panel: The panel.
            daily: The daily table.
            forecast_end: Last day to forecast.

        Returns:
            (daily forecast per meter, fitted model, issue day).

        Raises:
            Errors.Misc.HorizonExceeded: When the forecast needs more days than the model was trained for.
        """

    @abstractmethod
    def write_forecast(self, daily: pd.DataFrame, future: pd.DataFrame, model, issue: pd.Timestamp,
                       forecast_end) -> None:
        """
        #Feature: Forecasting
        #Scenario: Save the year-end forecast by meter, zone and facility, plus the monthly outlook
        #Complexity: Time O(n) - rows

        Args:
            daily: The daily table.
            future: Year-end forecast per meter.
            model: The fitted model.
            issue: Issue day.
            forecast_end: Last forecast day.

        Raises:
            ValueError: When an argument is None.
        """

    @abstractmethod
    def long_range_check(self, panel, daily: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        #Feature: Forecasting
        #Scenario: How accurate were facility forecasts 1-30, 31-60 and 61-90 days ahead?
        #Complexity: Time O(months * fit) - one model per month

        Args:
            panel: The panel.
            daily: The daily table.

        Returns:
            (detail per issue month and window, overall accuracy per window).

        Raises:
            ValueError: When an argument is None.
        """

    @abstractmethod
    def read_forecast(self, level: str, start: Optional[str], end: Optional[str], page: int, limit: int) -> Dict:
        """
        #Feature: Forecasting
        #Scenario: Page through the saved forecast
        #Complexity: Time O(n) - rows

        Args:
            level: "meters", "zones" or "facility".
            start: Optional first date (YYYY-MM-DD).
            end: Optional last date (YYYY-MM-DD).
            page: 1-based page.
            limit: Rows per page.

        Returns:
            Dict with total, page, limit and rows.

        Raises:
            Errors.Misc.DataNotReady: When the pipeline has not run yet.
        """
