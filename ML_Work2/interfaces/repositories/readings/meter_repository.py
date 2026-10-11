"""Interface for reading meter reference data from the warehouse."""
from abc import ABC, abstractmethod

import pandas as pd


class IMeterRepository(ABC):
    """
    #Feature: Meter reference data
    #Scenario: Services depend on this interface, never on a database
    #Complexity: Time O(m) - meters

    Meter and breaker reference data.
    """

    @abstractmethod
    def get_meters(self) -> pd.DataFrame:
        """
        #Feature: Meter reference data
        #Scenario: Every canonical meter with its zone
        #Complexity: Time O(m) - meters

        Returns:
            Columns meter_key, meter_label, zone_name, zone_latitude, zone_longitude.

        Raises:
            Errors.HTTP.InternalServerError: When the warehouse query fails.
        """

    @abstractmethod
    def get_breakers(self) -> pd.DataFrame:
        """
        #Feature: Meter reference data
        #Scenario: Circuit-breaker rating of every meter
        #Complexity: Time O(m) - meters

        Returns:
            Columns meter_key, meter_label, cb_max_rating_a (NaN when none recorded).

        Raises:
            Errors.HTTP.InternalServerError: When the warehouse query fails.
        """
