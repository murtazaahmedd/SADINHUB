"""Interface for observed daily temperature."""
from abc import ABC, abstractmethod
from datetime import date

import pandas as pd


class IWeatherRepository(ABC):
    """
    #Feature: Weather source
    #Scenario: Services read weather through this interface
    #Complexity: Time O(d) - days

    Observed daily temperature for a site.
    """

    @abstractmethod
    def fetch_archive(self, latitude: float, longitude: float, start: date, end: date) -> pd.DataFrame:
        """
        #Feature: Weather source
        #Scenario: Reanalysis archive temperatures for a date range
        #Complexity: Time O(d) - days

        Args:
            latitude: Site latitude.
            longitude: Site longitude.
            start: First day.
            end: Last day.

        Returns:
            Columns date, t_max, t_min, t_mean.

        Raises:
            Errors.HTTP.InternalServerError: When the source cannot be reached.
        """

    @abstractmethod
    def fetch_recent(self, latitude: float, longitude: float, start: date, end: date) -> pd.DataFrame:
        """
        #Feature: Weather source
        #Scenario: The last few days the archive lags behind real time (observations, not forecasts)
        #Complexity: Time O(d) - days

        Args:
            latitude: Site latitude.
            longitude: Site longitude.
            start: First day.
            end: Last day.

        Returns:
            Columns date, t_max, t_min, t_mean.

        Raises:
            Errors.HTTP.InternalServerError: When the source cannot be reached.
        """
