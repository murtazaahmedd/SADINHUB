"""Interface for scoring forecasts against actuals."""
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import pandas as pd


class IAccuracyService(ABC):
    """
    #Feature: Accuracy
    #Scenario: Callers depend on this interface, never on the scoring helpers
    #Complexity: Time O(n) - scored periods

    Scores every forecast against gold-booked actuals and saves the results.
    """

    @abstractmethod
    def score(self, panel, preds: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        #Feature: Accuracy
        #Scenario: Score meter, zone and facility forecasts at every horizon
        #Complexity: Time O(n) - scored periods

        Args:
            panel: The panel.
            preds: Every-morning forecasts.

        Returns:
            (per-meter periods, rows for every level).

        Raises:
            ValueError: When an argument is None.
        """

    @abstractmethod
    def write_validation(self, panel, preds: pd.DataFrame, periods: pd.DataFrame, levels: pd.DataFrame,
                         coefficients: Dict) -> None:
        """
        #Feature: Accuracy
        #Scenario: Save every accuracy table
        #Complexity: Time O(n) - scored periods

        Args:
            panel: The panel.
            preds: Every-morning forecasts.
            periods: Per-meter periods.
            levels: Rows for every level.
            coefficients: Learned weights of each monthly model.

        Raises:
            ValueError: When an argument is None.
        """

    @abstractmethod
    def read_summary(self) -> List[Dict]:
        """
        #Feature: Accuracy
        #Scenario: The saved accuracy summary
        #Complexity: Time O(1) - small file

        Returns:
            One dict per level and horizon.

        Raises:
            Errors.Misc.DataNotReady: When the pipeline has not run yet.
        """

    @abstractmethod
    def read_month_matrix(self, level: str, horizon: str) -> Dict:
        """
        #Feature: Accuracy
        #Scenario: The saved unit x month accuracy table
        #Complexity: Time O(n) - table size

        Args:
            level: "meter", "zone" or "facility".
            horizon: "1 day", "7 days", "30 days" or "calendar month".

        Returns:
            Dict with columns and rows.

        Raises:
            Errors.HTTP.BadRequest: When level or horizon is unknown.
            Errors.Misc.DataNotReady: When the pipeline has not run yet.
        """
