"""Open-Meteo observed daily temperature (free, keyless; only coordinates are sent)."""
import json
import urllib.request
from datetime import date

import pandas as pd

from constants.weather import ARCHIVE_URL, DAILY_FIELDS, FORECAST_URL, REQUEST_TIMEOUT_SECONDS
from errors import Errors
from interfaces.repositories.weather.weather_repository import IWeatherRepository
from utilities.guard import require, traced


class WeatherRepository(IWeatherRepository):
    """
    #Feature: Weather source
    #Scenario: Fetch observed temperatures from the Open-Meteo HTTP API
    #Complexity: Time O(d) - days

    ERA5 reanalysis archive plus the recent past from the forecast service.
    """

    @traced
    def _fetch(self, url: str, latitude: float, longitude: float, start: date, end: date) -> pd.DataFrame:
        """
        #Feature: Weather source
        #Scenario: One HTTP request for a date range
        #Complexity: Time O(d) - days

        Args:
            url: Archive or forecast endpoint.
            latitude: Site latitude.
            longitude: Site longitude.
            start: First day.
            end: Last day.

        Returns:
            Columns date, t_max, t_min, t_mean.

        Raises:
            ValueError: When an argument is None.
            Errors.HTTP.InternalServerError: When the request fails.
        """
        require("WeatherRepository._fetch", "url", url)
        require("WeatherRepository._fetch", "latitude", latitude)
        require("WeatherRepository._fetch", "longitude", longitude)
        query = (f"{url}?latitude={latitude}&longitude={longitude}&start_date={start}&end_date={end}"
                 f"&daily={DAILY_FIELDS}&timezone=Asia%2FRiyadh")
        try:
            with urllib.request.urlopen(query, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                daily = json.loads(response.read().decode())["daily"]
        except OSError as exc:
            raise Errors.HTTP.InternalServerError(
                f"[WeatherRepository._fetch] Weather request failed for {start}..{end}. Invariant violated: "
                f"Open-Meteo is reachable. Received: {exc}. Fix: check the internet connection and retry.") from exc
        return pd.DataFrame({"date": pd.to_datetime(daily["time"]), "t_max": daily["temperature_2m_max"],
                             "t_min": daily["temperature_2m_min"], "t_mean": daily["temperature_2m_mean"]})

    @traced
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
        return self._fetch(ARCHIVE_URL, latitude, longitude, start, end)

    @traced
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
        return self._fetch(FORECAST_URL, latitude, longitude, start, end)
