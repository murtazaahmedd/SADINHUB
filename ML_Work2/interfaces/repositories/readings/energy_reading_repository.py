"""Interface for reading raw cumulative kWh readings."""
from abc import ABC, abstractmethod

import pandas as pd


class IEnergyReadingRepository(ABC):
    """
    #Feature: Raw readings
    #Scenario: Services depend on this interface, never on a database
    #Complexity: Time O(n) - readings

    Raw meter readings.
    """

    @abstractmethod
    def get_readings(self, meter_key: str) -> pd.DataFrame:
        """
        #Feature: Raw readings
        #Scenario: Every cumulative-kWh reading of one meter, oldest first
        #Complexity: Time O(n) - readings

        Args:
            meter_key: The meter's key (uuid as text).

        Returns:
            Columns reading_timestamp_utc and active_energy_cumulative_kwh.

        Raises:
            ValueError: When meter_key is None.
            Errors.HTTP.InternalServerError: When the warehouse query fails.
        """
