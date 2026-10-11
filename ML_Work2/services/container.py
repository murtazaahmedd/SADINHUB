"""Lazy-loaded services, wired from the repository container (constructor injection)."""
from interfaces.services.accuracy.accuracy_service import IAccuracyService
from interfaces.services.anomaly.anomaly_service import IAnomalyService
from interfaces.services.billing.billing_service import IBillingService
from interfaces.services.dashboard.dashboard_service import IDashboardService
from interfaces.services.data.data_service import IDataService
from interfaces.services.data.weather_service import IWeatherService
from interfaces.services.forecast.forecast_service import IForecastService
from interfaces.services.pipeline.pipeline_service import IPipelineService
from repositories.container import RepositoryContainer
from services.accuracy.accuracy_service import AccuracyService
from services.anomaly.anomaly_service import AnomalyService
from services.billing.billing_service import BillingService
from services.dashboard.dashboard_service import DashboardService
from services.data.data_service import DataService
from services.data.weather_service import WeatherService
from services.forecast.forecast_service import ForecastService
from services.pipeline.pipeline_service import PipelineService
from utilities.guard import require, traced


class ServiceContainer:
    """
    #Feature: Dependency injection
    #Scenario: One place that creates every service, on first use
    #Complexity: Time O(1) - constant per property

    Holds the repository container and builds services lazily.
    """

    @traced
    def __init__(self, repositories: RepositoryContainer) -> None:
        """
        #Feature: Dependency injection
        #Scenario: Constructor injection of the repositories
        #Complexity: Time O(1) - constant

        Args:
            repositories: The repository container.

        Raises:
            ValueError: When repositories is None.
        """
        require("ServiceContainer.__init__", "repositories", repositories)
        self._repos = repositories
        self._data = self._weather = self._forecast = self._accuracy = None
        self._billing = self._dashboard = self._anomaly = self._pipeline = None

    @property
    def data(self) -> IDataService:
        """Meter data and the daily table."""
        if self._data is None:
            r = self._repos
            self._data = DataService(r.meters, r.energy_readings, r.excel_readings, r.gold_reference, r.artifacts)
        return self._data

    @property
    def weather(self) -> IWeatherService:
        """Observed temperature."""
        if self._weather is None:
            self._weather = WeatherService(self._repos.weather, self._repos.artifacts)
        return self._weather

    @property
    def forecast(self) -> IForecastService:
        """Training, replay and forecast."""
        if self._forecast is None:
            self._forecast = ForecastService(self._repos.artifacts)
        return self._forecast

    @property
    def accuracy(self) -> IAccuracyService:
        """Scoring."""
        if self._accuracy is None:
            self._accuracy = AccuracyService(self._repos.artifacts)
        return self._accuracy

    @property
    def billing(self) -> IBillingService:
        """Bill forecast."""
        if self._billing is None:
            self._billing = BillingService(self._repos.meters, self._repos.artifacts)
        return self._billing

    @property
    def dashboard(self) -> IDashboardService:
        """Dashboard page."""
        if self._dashboard is None:
            self._dashboard = DashboardService(self._repos.artifacts, self.billing)
        return self._dashboard

    @property
    def anomaly(self) -> IAnomalyService:
        """Consumption alerts and data-problem alerts."""
        if self._anomaly is None:
            self._anomaly = AnomalyService(self._repos.artifacts, self.dashboard, self._pipeline_results)
        return self._anomaly

    @property
    def pipeline(self) -> IPipelineService:
        """The whole run."""
        if self._pipeline is None:
            self._pipeline = PipelineService(self.data, self.weather, self.forecast, self.accuracy, self.billing,
                                             self.dashboard, self._repos.artifacts)
        return self._pipeline

    def _pipeline_results(self):
        """Callable handed to the anomaly service so it can rebuild the dashboard from the saved run."""
        return self.pipeline.saved_results()
