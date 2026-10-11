"""Meter and breaker reference data, read through api.* functions."""
import pandas as pd

from interfaces.database import ISessionFactory
from interfaces.repositories.readings.meter_repository import IMeterRepository
from repositories.readings.udf_query import call_udf
from utilities.guard import require, traced


class MeterRepository(IMeterRepository):
    """
    #Feature: Meter reference data
    #Scenario: Read meters and breaker ratings from the warehouse
    #Complexity: Time O(m) - meters

    Calls api.get_energy_meters() and api.get_meter_breakers().
    """

    @traced
    def __init__(self, sessions: ISessionFactory) -> None:
        """
        #Feature: Meter reference data
        #Scenario: Constructor injection of the connection source
        #Complexity: Time O(1) - constant

        Args:
            sessions: Source of read-only connections.

        Raises:
            ValueError: When sessions is None.
        """
        require("MeterRepository.__init__", "sessions", sessions)
        self._sessions = sessions

    @traced
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
        return call_udf(self._sessions, "SELECT * FROM api.get_energy_meters()")

    @traced
    def get_breakers(self) -> pd.DataFrame:
        """
        #Feature: Meter reference data
        #Scenario: Circuit-breaker rating of every meter
        #Complexity: Time O(m) - meters

        Returns:
            Columns meter_key, meter_label, cb_max_rating_a.

        Raises:
            Errors.HTTP.InternalServerError: When the warehouse query fails.
        """
        return call_udf(self._sessions, "SELECT * FROM api.get_meter_breakers()")
