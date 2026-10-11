"""The gold layer's own daily consumption, read through api.get_gold_daily_consumption()."""
import pandas as pd

from interfaces.database import ISessionFactory
from interfaces.repositories.readings.gold_reference_repository import IGoldReferenceRepository
from repositories.readings.udf_query import call_udf
from utilities.guard import require, traced


class GoldReferenceRepository(IGoldReferenceRepository):
    """
    #Feature: Gold reference
    #Scenario: Daily consumption as gold books it
    #Complexity: Time O(d) - meter-days

    Calls api.get_gold_daily_consumption().
    """

    @traced
    def __init__(self, sessions: ISessionFactory) -> None:
        """
        #Feature: Gold reference
        #Scenario: Constructor injection of the connection source
        #Complexity: Time O(1) - constant

        Args:
            sessions: Source of read-only connections.

        Raises:
            ValueError: When sessions is None.
        """
        require("GoldReferenceRepository.__init__", "sessions", sessions)
        self._sessions = sessions

    @traced
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
        return call_udf(self._sessions, "SELECT * FROM api.get_gold_daily_consumption()")
