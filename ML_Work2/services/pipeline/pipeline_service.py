"""End-to-end run: data -> validation of the year -> forecast to year end -> dashboard."""
import time
from datetime import date
from pathlib import Path
from typing import Callable, Dict, Optional

import pandas as pd

from config.settings import forecast_end_for, validation_start_for
from constants.forecasting import ALERT_HISTORY_WARMUP_MONTHS, WALK_FORWARD_DAYS
from interfaces.repositories.artifacts.artifact_repository import IArtifactRepository
from interfaces.services.accuracy.accuracy_service import IAccuracyService
from interfaces.services.billing.billing_service import IBillingService
from interfaces.services.dashboard.dashboard_service import IDashboardService
from interfaces.services.data.data_service import IDataService
from interfaces.services.data.weather_service import IWeatherService
from interfaces.services.forecast.forecast_service import IForecastService
from interfaces.services.pipeline.pipeline_service import IPipelineService
from models.results import PipelineResults
from utilities.guard import require, traced


@traced
def _long_range_since(detail: pd.DataFrame) -> str:
    """
    #Feature: Pipeline
    #Scenario: Month of the first long-range replay, as shown on the dashboard
    #Complexity: Time O(n) - rows

    Args:
        detail: Long-range accuracy detail with an "issued" column.

    Returns:
        Text such as "Sep 2025".

    Raises:
        ValueError: When detail is None.
    """
    require("_long_range_since", "detail", detail)
    return pd.Timestamp(detail["issued"].min()).strftime("%b %Y")


@traced
def _long_range_overall(detail: pd.DataFrame) -> Dict:
    """
    #Feature: Pipeline
    #Scenario: Overall accuracy per long-range window from the saved detail
    #Complexity: Time O(n) - rows

    Args:
        detail: Long-range accuracy detail.

    Returns:
        window -> accuracy percent.

    Raises:
        ValueError: When detail is None.
    """
    require("_long_range_overall", "detail", detail)
    return detail.groupby("window").apply(
        lambda g: round(100 * (1 - (g["forecast_kwh"] - g["actual_kwh"]).abs().sum() / g["actual_kwh"].sum()), 1),
        include_groups=False).to_dict()


class PipelineService(IPipelineService):
    """
    #Feature: Pipeline
    #Scenario: Orchestrate every stage through its service interface
    #Complexity: Time O(months * fit) - one model per month

    Wires data, weather, forecast, accuracy, billing and dashboard services.
    """

    @traced
    def __init__(self, data: IDataService, weather: IWeatherService, forecast: IForecastService,
                 accuracy: IAccuracyService, billing: IBillingService, dashboard: IDashboardService,
                 artifacts: IArtifactRepository) -> None:
        """
        #Feature: Pipeline
        #Scenario: Constructor injection of every stage
        #Complexity: Time O(1) - constant

        Args:
            data: Data service.
            weather: Weather service.
            forecast: Forecast service.
            accuracy: Accuracy service.
            billing: Billing service.
            dashboard: Dashboard service.
            artifacts: Saved-results store.

        Raises:
            ValueError: When an argument is None.
        """
        for name, value in (("data", data), ("weather", weather), ("forecast", forecast), ("accuracy", accuracy),
                            ("billing", billing), ("dashboard", dashboard), ("artifacts", artifacts)):
            require("PipelineService.__init__", name, value)
        self._data, self._weather, self._forecast = data, weather, forecast
        self._accuracy, self._billing, self._dashboard = accuracy, billing, dashboard
        self._artifacts = artifacts

    @traced
    def _site(self) -> tuple:
        """
        #Feature: Pipeline
        #Scenario: Site coordinates taken from the meters' zones
        #Complexity: Time O(m) - meters

        Returns:
            (latitude, longitude) rounded to 8 decimals.

        Raises:
            ValueError: When the meters disagree on location.
        """
        meters = self._data.load_meters()
        lat, lon = round(float(meters["zone_latitude"].mean()), 8), round(float(meters["zone_longitude"].mean()), 8)
        if meters["zone_latitude"].nunique() > 1 or meters["zone_longitude"].nunique() > 1:
            raise ValueError(f"[PipelineService._site] Meters are at more than one location. Invariant violated: one "
                             f"site. Received: {meters[['zone_latitude', 'zone_longitude']].drop_duplicates().shape[0]} "
                             f"locations. Fix: run one pipeline per site.")
        return lat, lon

    @traced
    def _load_inputs(self, daily: pd.DataFrame) -> tuple:
        """
        #Feature: Pipeline
        #Scenario: Weather, panel and the year's boundary dates, derived from the data
        #Complexity: Time O(m*n) - meters times days

        Args:
            daily: The daily table.

        Returns:
            (panel, last complete day, forecast end, validation start).

        Raises:
            ValueError: When daily is None.
        """
        require("PipelineService._load_inputs", "daily", daily)
        last_complete = daily.loc[daily["complete_day"], "date"].max()
        end, start = forecast_end_for(last_complete.date()), validation_start_for(last_complete.date())
        lat, lon = self._site()
        weather = self._weather.load_weather(daily["date"].max(), lat, lon)
        return self._forecast.build_panel(daily, weather, end), last_complete, end, start

    @traced
    def _alert_history(self, panel, preds: pd.DataFrame, start: date, say) -> None:
        """
        #Feature: Pipeline
        #Scenario: Same-day forecasts for the months before the validation year, so the alerts train on a full seasonal cycle
        #Complexity: Time O(months * fit) - one model per extra month

        Each early morning is forecast by a model trained only on data before that month, exactly like the
        validation year. Only the same-day forecast is needed, so the replay is cheap.

        Args:
            panel: The panel.
            preds: Every-morning forecasts of the validation year.
            start: First day of the validation year.
            say: Progress callback.

        Raises:
            ValueError: When panel is None.
        """
        require("PipelineService._alert_history", "panel", panel)
        first = (panel.dates[0].to_period("M") + ALERT_HISTORY_WARMUP_MONTHS).to_timestamp()
        if first >= pd.Timestamp(start):
            self._artifacts.save_frame("every_morning_forecast_extended", preds)
            return
        say(f"    same-day forecasts from {first:%d %b %Y} for the alerts' training history")
        early, _ = self._forecast.walk_forward(panel, first, pd.Timestamp(start) - pd.Timedelta(days=1), 1, say)
        self._artifacts.save_frame("every_morning_forecast_extended", pd.concat([early, preds], ignore_index=True))

    @traced
    def run(self, refresh_sources: bool = False, log: Optional[Callable[[str], None]] = None) -> PipelineResults:
        """
        #Feature: Pipeline
        #Scenario: Run every stage and save every result
        #Complexity: Time O(months * fit) - one model per month

        Args:
            refresh_sources: Re-read the warehouse and the Excel exports.
            log: Progress callback.

        Returns:
            A PipelineResults bundle.

        Raises:
            Errors.Misc.MeterMappingError: When an Excel export cannot be tied to one meter.
            Errors.Misc.HorizonExceeded: When the forecast needs more days than the model was trained for.
        """
        say = log or (lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True))
        say("1/5 Reading meter data (database + Excel) and building daily kWh")
        daily = self._data.build_daily_actuals(refresh=True, refresh_sources=refresh_sources)
        say(f"    {daily['meter_label'].nunique()} meters, data to "
            f"{daily.loc[daily['complete_day'], 'date'].max():%Y-%m-%d}")
        say("2/5 Riyadh temperature history and seasonal normals")
        panel, last_complete, end, start = self._load_inputs(daily)
        say("3/5 Replaying every morning of the year (walk-forward validation)")
        preds, coefficients = self._forecast.walk_forward(panel, start, last_complete, WALK_FORWARD_DAYS, say)
        self._artifacts.save_frame("every_morning_forecast", preds)
        self._alert_history(panel, preds, start, say)
        say("4/5 Scoring against gold actuals")
        periods, levels = self._accuracy.score(panel, preds)
        self._accuracy.write_validation(panel, preds, periods, levels, coefficients)
        say(f"5/5 Forecasting to {end:%d %B %Y} (and checking how far ahead it stays accurate)")
        return self._forecast_stage(panel, daily, preds, levels, end, start, say)

    @traced
    def _forecast_stage(self, panel, daily, preds, levels, end: date, start: date, say) -> PipelineResults:
        """
        #Feature: Pipeline
        #Scenario: Year-end forecast, long-range check and breaker table
        #Complexity: Time O(fit + months * fit) - year-end fit plus monthly checks

        Args:
            panel: The panel.
            daily: The daily table.
            preds: Every-morning forecasts.
            levels: Scored rows for every level.
            end: Last forecast day.
            start: First replayed day.
            say: Progress callback.

        Returns:
            A PipelineResults bundle.

        Raises:
            Errors.Misc.HorizonExceeded: When the forecast needs more days than the model was trained for.
        """
        future, model, issue = self._forecast.forecast_to_year_end(panel, daily, end)
        self._forecast.write_forecast(daily, future, model, issue, end)
        detail, overall = self._forecast.long_range_check(panel, daily)
        self._artifacts.save_frame("long_range_accuracy", detail)
        say(f"    facility accuracy by distance ahead: {overall}")
        self._billing.load_breakers(refresh=True)
        say("Done.")
        return PipelineResults(panel, daily, preds, levels, future, issue, overall, end, start,
                               _long_range_since(detail))

    @traced
    def saved_results(self) -> PipelineResults:
        """
        #Feature: Pipeline
        #Scenario: Reload the last full run without retraining
        #Complexity: Time O(n) - saved files

        Returns:
            A PipelineResults bundle.

        Raises:
            Errors.Misc.DataNotReady: When the pipeline has not run yet.
        """
        daily = self._data.build_daily_actuals()
        panel, _, end, start = self._load_inputs(daily)
        preds = self._artifacts.load_frame("every_morning_forecast")
        _, levels = self._accuracy.score(panel, preds)
        future = self._artifacts.load_frame("forecast_meters", parse_dates=["date"])
        info = self._artifacts.load_json("forecast_info")
        detail = self._artifacts.load_frame("long_range_accuracy")
        return PipelineResults(panel, daily, preds, levels, future, pd.Timestamp(info["forecast_issued_for_morning_of"]),
                               _long_range_overall(detail), end, start, _long_range_since(detail))

    @traced
    def build_dashboard(self, results: Optional[PipelineResults] = None) -> Path:
        """
        #Feature: Pipeline
        #Scenario: Build the dashboard page from a run, or from the last saved run
        #Complexity: Time O(n) - page data

        Args:
            results: Optional PipelineResults; the saved run is used when omitted.

        Returns:
            Path of the HTML page.

        Raises:
            Errors.Misc.DataNotReady: When there is no saved run.
        """
        return self._dashboard.build(results or self.saved_results())

    @traced
    def status(self) -> Dict[str, bool]:
        """
        #Feature: Pipeline
        #Scenario: Which saved results exist yet?
        #Complexity: Time O(k) - number of results checked

        Returns:
            result name -> whether its file exists.

        Raises:
            None.
        """
        keys = ("daily_actuals", "every_morning_forecast", "forecast_meters", "accuracy_summary_json",
                "bill_forecast", "alerts_consumption", "dashboard")
        return {key: self._artifacts.exists(key) for key in keys}
