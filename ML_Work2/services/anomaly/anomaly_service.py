"""Compute the consumption alerts and rebuild the dashboard with the Alerts tab.

Reads the saved forecasting outputs (never modifies them) and writes only to
outputs/alerts/. No database connection is used. Run it again after every
forecasting run, so the alerts use the latest forecasts; the Alerts tab shows a
notice if they are out of date.
"""
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from constants.known_events import KNOWN_REGISTER_JUMP
from interfaces.repositories.artifacts.artifact_repository import IArtifactRepository
from interfaces.services.anomaly.anomaly_service import IAnomalyService
from interfaces.services.dashboard.dashboard_service import IDashboardService
from models.anomaly.detectors import detect, period_plan
from models.anomaly.features import build_features
from models.anomaly.shape import night_alerts, night_scores
from utilities.anomaly_incidents import build_incidents
from utilities.anomaly_inputs import build_meter_days
from utilities.anomaly_quality import historical_register_check, quality_alerts
from models.anomaly.evaluation import evaluate
from utilities.anomaly_report import alert_table, dashboard_block, extras_summary, incident_rows, night_rows
from utilities.anomaly_shape import night_base_load
from utilities.guard import require, traced

_SCORED_COLUMNS = ["meter_label", "zone", "date", "period", "eligible", "history_ok", "expected_kwh", "actual_kwh",
                   "diff_kwh", "usual_error_kwh", "z", "rel", "jump", "similar", "level", "ml_score", "material",
                   "rule_flag", "ml_flag", "detector"]


class AnomalyService(IAnomalyService):
    """
    #Feature: Alerts
    #Scenario: Detect, save and serve alerts
    #Complexity: Time O(n * trees) - meter-days times forest size

    Wires the detectors, the store and the dashboard.
    """

    @traced
    def __init__(self, artifacts: IArtifactRepository, dashboard: IDashboardService,
                 results_provider: Callable[[], object]) -> None:
        """
        #Feature: Alerts
        #Scenario: Constructor injection of the store, the dashboard and the saved-run loader
        #Complexity: Time O(1) - constant

        Args:
            artifacts: Saved-results store.
            dashboard: Dashboard service (rebuilt after the alerts change).
            results_provider: Callable returning the last saved PipelineResults.

        Raises:
            ValueError: When an argument is None.
        """
        require("AnomalyService.__init__", "artifacts", artifacts)
        require("AnomalyService.__init__", "dashboard", dashboard)
        require("AnomalyService.__init__", "results_provider", results_provider)
        self._artifacts = artifacts
        self._dashboard = dashboard
        self._results = results_provider

    @traced
    def _night(self, scored: pd.DataFrame, intervals: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        #Feature: Alerts
        #Scenario: Night base-load alerts on the days the daily detectors could judge
        #Complexity: Time O(n) - five-minute intervals

        The night series starts before the replay (so a baseline exists on day one); alerts are kept only
        for trusted days inside the replay, with their zone and period.

        Args:
            scored: Scored meter-day rows.
            intervals: Validated five-minute intervals.

        Returns:
            (night alerts with zone and period, number of scored nights per period).

        Raises:
            ValueError: When an argument is None.
        """
        require("AnomalyService._night", "scored", scored)
        scores = night_scores(night_base_load(intervals))
        gate = scored.loc[scored["eligible"], ["meter_label", "date", "zone", "period"]]
        judged = scores.merge(gate, on=["meter_label", "date"])
        return night_alerts(judged), judged.groupby("period").size()

    @traced
    def _save(self, scored: pd.DataFrame, alerts: pd.DataFrame, quality: pd.DataFrame, evaluation: Dict,
              block: Dict, others: Dict[str, pd.DataFrame]) -> None:
        """
        #Feature: Alerts
        #Scenario: Write every alert output
        #Complexity: Time O(n) - rows

        Args:
            scored: Scored rows.
            alerts: Consumption alerts.
            quality: Data-problem episodes.
            evaluation: Evaluation dict.
            block: The dashboard block.
            others: incidents and night-alert tables.

        Raises:
            ValueError: When an argument is None.
        """
        require("AnomalyService._save", "scored", scored)
        self._artifacts.save_frame("alerts_scored", scored[_SCORED_COLUMNS])
        self._artifacts.save_frame("alerts_consumption", alerts)
        self._artifacts.save_frame("alerts_data_problems", quality)
        for key, frame in (("alerts_incidents", others["incidents"]), ("alerts_night", others["night"])):
            self._artifacts.save_frame(key, frame)
        self._artifacts.save_json("alerts_evaluation", evaluation, indent=2, default=str)
        self._artifacts.save_json("alerts_dashboard", block, separators=(",", ":"), default=str)

    @traced
    def _detect(self, daily: pd.DataFrame, history: str) -> Tuple[pd.DataFrame, object, Dict, List, pd.DataFrame, Dict]:
        """
        #Feature: Alerts
        #Scenario: Run every detector on the saved forecasts
        #Complexity: Time O(n * trees) - meter-days times forest size

        Args:
            daily: The daily table.
            history: Key of the same-day forecast file to use.

        Returns:
            (scored rows, forest, settings, period plan, intervals, frame attrs).

        Raises:
            Errors.Misc.InsufficientData: When the development period has nothing to train on.
        """
        intervals = self._artifacts.load_frame("validated_intervals")
        frame = build_meter_days(daily, self._artifacts.load_frame(history), intervals)
        plan = period_plan(int(daily.loc[daily["complete_day"], "date"].max().year), frame["date"].min().date())
        scored, forest, settings = detect(build_features(frame, daily), plan)
        return scored, forest, settings, plan, intervals, frame.attrs

    @traced
    def run(self) -> Dict:
        """
        #Feature: Alerts
        #Scenario: Score every forecast day, save the alerts and rebuild the dashboard's Alerts tab
        #Complexity: Time O(n * trees) - meter-days times forest size

        Returns:
            Alerts per period, the final-test agreement and what incidents and night load add.

        Raises:
            Errors.Misc.DataNotReady: When the forecasting pipeline has not run yet.
            Errors.Misc.InsufficientData: When the development period has nothing to train on.
        """
        daily = self._artifacts.load_frame("daily_actuals")
        history = "every_morning_forecast_extended" if self._artifacts.exists("every_morning_forecast_extended") \
            else "every_morning_forecast"
        scored, forest, settings, plan, intervals, attrs = self._detect(daily, history)
        quality, alerts = quality_alerts(scored), alert_table(scored)
        night, night_checked = self._night(scored, intervals)
        incidents = build_incidents(alerts, scored["date"].max())
        evaluation = evaluate(scored, quality, forest, settings, historical_register_check(daily, *KNOWN_REGISTER_JUMP), plan)
        evaluation["mapping_rows_checked"] = attrs["mapping_rows_checked"]
        evaluation["extras"] = extras_summary(alerts, incidents, night, night_checked, plan)
        inputs = {name: self._artifacts.fingerprint(name) for name in ("daily_actuals", history)}
        extras = {"incidents": incident_rows(incidents), "night": night_rows(night)}
        self._save(scored, alerts, quality, evaluation, dashboard_block(scored, alerts, quality, evaluation, inputs, extras),
                   {"incidents": incidents, "night": night})
        self._dashboard.build(self._results())
        return {"periods": evaluation["periods"], "final_test_agreement": evaluation["final_test_agreement"],
                "extras": evaluation["extras"]}

    @traced
    def read_alerts(self, period: Optional[str], detector: Optional[str], meter: Optional[str],
                    page: int, limit: int) -> Dict:
        """
        #Feature: Alerts
        #Scenario: Page through the saved consumption alerts
        #Complexity: Time O(n) - alerts

        Args:
            period: Optional period name.
            detector: Optional "rule", "ml" or "both".
            meter: Optional meter label.
            page: 1-based page.
            limit: Rows per page.

        Returns:
            Dict with total, page, limit and rows.

        Raises:
            Errors.Misc.DataNotReady: When the alerts have not been computed yet.
        """
        frame = self._artifacts.load_frame("alerts_consumption")
        for column, value in (("period", period), ("detector", detector), ("meter_label", meter)):
            if value:
                frame = frame[frame[column] == value]
        rows = frame.iloc[(page - 1) * limit:page * limit].replace({np.nan: None})
        return {"total": int(len(frame)), "page": page, "limit": limit, "rows": rows.to_dict("records")}

    @traced
    def read_data_problems(self) -> Dict:
        """
        #Feature: Alerts
        #Scenario: The saved data-problem episodes
        #Complexity: Time O(n) - episodes

        Returns:
            Dict with total and rows.

        Raises:
            Errors.Misc.DataNotReady: When the alerts have not been computed yet.
        """
        frame = self._artifacts.load_frame("alerts_data_problems")
        return {"total": int(len(frame)), "rows": frame.replace({np.nan: None}).to_dict("records")}

    @traced
    def read_incidents(self, active_only: bool, page: int, limit: int) -> Dict:
        """
        #Feature: Alerts
        #Scenario: Page through the saved incidents, biggest impact first
        #Complexity: Time O(n) - incidents

        Args:
            active_only: Keep only incidents whose last alert is within two days of the data end.
            page: 1-based page.
            limit: Rows per page.

        Returns:
            Dict with total, page, limit and rows.

        Raises:
            Errors.Misc.DataNotReady: When the alerts have not been computed yet.
        """
        frame = self._artifacts.load_frame("alerts_incidents")
        frame = frame[frame["active"]] if active_only else frame
        rows = frame.iloc[(page - 1) * limit:page * limit].replace({np.nan: None})
        return {"total": int(len(frame)), "page": page, "limit": limit, "rows": rows.to_dict("records")}

    @traced
    def read_night_alerts(self) -> Dict:
        """
        #Feature: Alerts
        #Scenario: The saved night base-load alerts
        #Complexity: Time O(n) - alerts

        Returns:
            Dict with total and rows.

        Raises:
            Errors.Misc.DataNotReady: When the alerts have not been computed yet.
        """
        frame = self._artifacts.load_frame("alerts_night")
        return {"total": int(len(frame)), "rows": frame.replace({np.nan: None}).to_dict("records")}
