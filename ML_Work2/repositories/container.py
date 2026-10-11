"""Lazy-loaded repositories. Nothing connects to a database until a query actually runs."""
from typing import Optional

from config.settings import Settings
from database.session import ReadOnlySessionFactory
from interfaces.database import ISessionFactory
from interfaces.repositories.artifacts.artifact_repository import IArtifactRepository
from interfaces.repositories.readings.energy_reading_repository import IEnergyReadingRepository
from interfaces.repositories.readings.excel_reading_repository import IExcelReadingRepository
from interfaces.repositories.readings.gold_reference_repository import IGoldReferenceRepository
from interfaces.repositories.readings.meter_repository import IMeterRepository
from interfaces.repositories.weather.weather_repository import IWeatherRepository
from repositories.artifacts.artifact_repository import ArtifactRepository
from repositories.readings.energy_reading_repository import EnergyReadingRepository
from repositories.readings.excel_reading_repository import ExcelReadingRepository
from repositories.readings.gold_reference_repository import GoldReferenceRepository
from repositories.readings.meter_repository import MeterRepository
from repositories.weather.weather_repository import WeatherRepository
from utilities.guard import require, traced


class RepositoryContainer:
    """
    #Feature: Dependency injection
    #Scenario: One place that creates every repository, on first use
    #Complexity: Time O(1) - constant per property

    Holds the settings and the connection source, and builds repositories lazily.
    """

    @traced
    def __init__(self, settings: Settings, sessions: Optional[ISessionFactory] = None) -> None:
        """
        #Feature: Dependency injection
        #Scenario: Constructor injection of the settings (and optionally a test connection source)
        #Complexity: Time O(1) - constant

        Args:
            settings: Folders and connection settings.
            sessions: Optional connection source (defaults to read-only Postgres).

        Raises:
            ValueError: When settings is None.
        """
        require("RepositoryContainer.__init__", "settings", settings)
        self._settings = settings
        self._sessions = sessions
        self._meters = self._readings = self._excel = self._gold = self._weather = self._artifacts = None

    @property
    def sessions(self) -> ISessionFactory:
        """Read-only connection source, created on first use."""
        if self._sessions is None:
            self._sessions = ReadOnlySessionFactory(self._settings)
        return self._sessions

    @property
    def meters(self) -> IMeterRepository:
        """Meter and breaker reference data."""
        if self._meters is None:
            self._meters = MeterRepository(self.sessions)
        return self._meters

    @property
    def energy_readings(self) -> IEnergyReadingRepository:
        """Raw warehouse readings."""
        if self._readings is None:
            self._readings = EnergyReadingRepository(self.sessions)
        return self._readings

    @property
    def excel_readings(self) -> IExcelReadingRepository:
        """Telemetry Excel exports."""
        if self._excel is None:
            self._excel = ExcelReadingRepository(self._settings.input_excel)
        return self._excel

    @property
    def gold_reference(self) -> IGoldReferenceRepository:
        """Gold-layer daily consumption, for checking our rule."""
        if self._gold is None:
            self._gold = GoldReferenceRepository(self.sessions)
        return self._gold

    @property
    def weather(self) -> IWeatherRepository:
        """Open-Meteo weather source."""
        if self._weather is None:
            self._weather = WeatherRepository()
        return self._weather

    @property
    def artifacts(self) -> IArtifactRepository:
        """Saved results."""
        if self._artifacts is None:
            self._artifacts = ArtifactRepository(self._settings.outputs)
        return self._artifacts
