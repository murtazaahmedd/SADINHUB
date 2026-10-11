"""Interface for the bill forecast."""
from abc import ABC, abstractmethod
from typing import Dict, Optional

import pandas as pd


class IBillingService(ABC):
    """
    #Feature: Bill forecast
    #Scenario: Callers depend on this interface, never on the tariff helpers
    #Complexity: Time O(m*c) - meters times cycles

    Converts kWh forecasts into SAR bills per meter and 30-day cycle.
    """

    @abstractmethod
    def load_breakers(self, refresh: bool = False) -> pd.DataFrame:
        """
        #Feature: Bill forecast
        #Scenario: Breaker rating, service class and monthly fee per meter
        #Complexity: Time O(m) - meters

        Args:
            refresh: Re-read from the warehouse instead of the saved copy.

        Returns:
            Columns meter_key, meter_label, breaker_a, service_class, service_fee_sar.

        Raises:
            Errors.Misc.InsufficientData: When a meter has no breaker rating.
        """

    @abstractmethod
    def build_bills(self, panel, daily: pd.DataFrame, preds: pd.DataFrame, future: pd.DataFrame,
                    issue: pd.Timestamp, breakers: pd.DataFrame, forecast_end) -> pd.DataFrame:
        """
        #Feature: Bill forecast
        #Scenario: One row per meter per 30-day cycle with predicted (and actual) bill
        #Complexity: Time O(m*c) - meters times cycles

        Args:
            panel: The panel.
            daily: The daily table.
            preds: Every-morning forecasts.
            future: Year-end forecast.
            issue: Issue day of the year-end forecast.
            breakers: Output of load_breakers.
            forecast_end: Last forecast day.

        Returns:
            The bill table (also saved).

        Raises:
            Errors.Misc.InsufficientData: When a meter has no breaker rating.
        """

    @abstractmethod
    def read_bills(self, cycle: Optional[int]) -> Dict:
        """
        #Feature: Bill forecast
        #Scenario: The saved bill forecast, optionally one cycle
        #Complexity: Time O(n) - rows

        Args:
            cycle: Optional cycle number.

        Returns:
            Dict with total and rows.

        Raises:
            Errors.Misc.DataNotReady: When the pipeline has not run yet.
        """
