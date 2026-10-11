"""Interface for consumption alerts and data-problem alerts."""
from abc import ABC, abstractmethod
from typing import Dict, Optional


class IAnomalyService(ABC):
    """
    #Feature: Alerts
    #Scenario: Callers depend on this interface, never on the detectors
    #Complexity: Time O(n * trees) - meter-days times forest size

    Flags unusual meter-days and days whose readings cannot be trusted.
    """

    @abstractmethod
    def run(self) -> Dict:
        """
        #Feature: Alerts
        #Scenario: Score every forecast day, save the alerts and rebuild the dashboard's Alerts tab
        #Complexity: Time O(n * trees) - meter-days times forest size

        Returns:
            Alerts per period and the final-test agreement.

        Raises:
            Errors.Misc.DataNotReady: When the forecasting pipeline has not run yet.
            Errors.Misc.InsufficientData: When the development period has nothing to train on.
        """

    @abstractmethod
    def read_alerts(self, period: Optional[str], detector: Optional[str], meter: Optional[str],
                    page: int, limit: int) -> Dict:
        """
        #Feature: Alerts
        #Scenario: Page through the saved consumption alerts
        #Complexity: Time O(n) - alerts

        Args:
            period: Optional period name.
            detector: Optional "rule", "ml" or "both".
            meter: Optional meter label.
            page: 1-based page.
            limit: Rows per page.

        Returns:
            Dict with total, page, limit and rows.

        Raises:
            Errors.Misc.DataNotReady: When the alerts have not been computed yet.
        """

    @abstractmethod
    def read_data_problems(self) -> Dict:
        """
        #Feature: Alerts
        #Scenario: The saved data-problem episodes
        #Complexity: Time O(n) - episodes

        Returns:
            Dict with total and rows.

        Raises:
            Errors.Misc.DataNotReady: When the alerts have not been computed yet.
        """

    @abstractmethod
    def read_incidents(self, active_only: bool, page: int, limit: int) -> Dict:
        """
        #Feature: Alerts
        #Scenario: Page through the saved incidents, biggest impact first
        #Complexity: Time O(n) - incidents

        Args:
            active_only: Keep only incidents whose last alert is within two days of the data end.
            page: 1-based page.
            limit: Rows per page.

        Returns:
            Dict with total, page, limit and rows.

        Raises:
            Errors.Misc.DataNotReady: When the alerts have not been computed yet.
        """

    @abstractmethod
    def read_night_alerts(self) -> Dict:
        """
        #Feature: Alerts
        #Scenario: The saved night base-load alerts
        #Complexity: Time O(n) - alerts

        Returns:
            Dict with total and rows.

        Raises:
            Errors.Misc.DataNotReady: When the alerts have not been computed yet.
        """
