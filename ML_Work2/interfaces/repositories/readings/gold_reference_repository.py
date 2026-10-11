"""Interface for the gold layer's own daily consumption (used only to check our reproduction)."""
from abc import ABC, abstractmethod

import pandas as pd


class IGoldReferenceRepository(ABC):
    """
    #Feature: Gold reference
    #Scenario: Compare our consumption rule with the gold layer
    #Complexity: Time O(d) - meter-days

    Daily consumption as the gold layer books it.
    """

    @abstractmethod
    def get_daily_consumption(self) -> pd.DataFrame:
        """
        #Feature: Gold reference
        #Scenario: Daily validated kWh per meter as gold books it
        #Complexity: Time O(d) - meter-days

        Returns:
            Columns meter_label, reading_date and gold_kwh.

        Raises:
            Errors.HTTP.InternalServerError: When the warehouse query fails.
        """
