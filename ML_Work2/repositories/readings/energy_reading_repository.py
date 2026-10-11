"""Raw cumulative kWh readings, read through api.get_energy_readings()."""
import pandas as pd

from interfaces.database import ISessionFactory
from interfaces.repositories.readings.energy_reading_repository import IEnergyReadingRepository
from repositories.readings.udf_query import call_udf
from utilities.guard import require, traced


class EnergyReadingRepository(IEnergyReadingRepository):
    """
    #Feature: Raw readings
    #Scenario: One meter's readings from the warehouse
    #Complexity: Time O(n) - readings

    Calls api.get_energy_readings(meter_key).
    """

    @traced
    def __init__(self, sessions: ISessionFactory) -> None:
        """
        #Feature: Raw readings
        #Scenario: Constructor injection of the connection source
        #Complexity: Time O(1) - constant

        Args:
            sessions: Source of read-only connections.

        Raises:
            ValueError: When sessions is None.
        """
        require("EnergyReadingRepository.__init__", "sessions", sessions)
        self._sessions = sessions

    @traced
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
        require("EnergyReadingRepository.get_readings", "meter_key", meter_key, str)
        return call_udf(self._sessions, "SELECT * FROM api.get_energy_readings(CAST(:key AS uuid))",
                        {"key": meter_key})
