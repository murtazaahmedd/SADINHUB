"""Compare forecasts with gold-booked actuals, save the accuracy tables and serve them back."""
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from constants.artifacts import ACCURACY_MONTH_FOLDER
from constants.forecasting import HORIZON_LABELS
from errors import Errors
from interfaces.repositories.artifacts.artifact_repository import IArtifactRepository
from interfaces.services.accuracy.accuracy_service import IAccuracyService
from utilities.guard import require, traced
from utilities.scoring import all_levels, calendar_months, meter_periods, month_matrix, summary

_LEVELS = ("meter", "zone", "facility")
_DAY_COUNT = {"1 day": 1, "7 days": 7, "30 days": 30}


class AccuracyService(IAccuracyService):
    """
    #Feature: Accuracy
    #Scenario: Score forecasts, write the tables, read them back for the API
    #Complexity: Time O(n) - scored periods

    Wraps utilities.scoring and the saved-results store.
    """

    @traced
    def __init__(self, artifacts: IArtifactRepository) -> None:
        """
        #Feature: Accuracy
        #Scenario: Constructor injection of the store
        #Complexity: Time O(1) - constant

        Args:
            artifacts: Saved-results store.

        Raises:
            ValueError: When artifacts is None.
        """
        require("AccuracyService.__init__", "artifacts", artifacts)
        self._artifacts = artifacts

    @traced
    def score(self, panel, preds: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        #Feature: Accuracy
        #Scenario: Score meter, zone and facility forecasts at every horizon
        #Complexity: Time O(n) - scored periods

        Args:
            panel: The panel.
            preds: Every-morning forecasts.

        Returns:
            (per-meter periods, rows for every level).

        Raises:
            ValueError: When an argument is None.
        """
        require("AccuracyService.score", "panel", panel)
        require("AccuracyService.score", "preds", preds)
        periods = pd.concat([meter_periods(panel, preds), calendar_months(panel, preds)], ignore_index=True)
        periods["horizon"] = periods["horizon"].map(HORIZON_LABELS)
        return periods, all_levels(periods)

    @traced
    def _write_detail(self, levels: pd.DataFrame) -> None:
        """
        #Feature: Accuracy
        #Scenario: Every scored forecast with its period dates
        #Complexity: Time O(n) - scored periods

        Args:
            levels: Rows for every level.

        Raises:
            ValueError: When levels is None.
        """
        require("AccuracyService._write_detail", "levels", levels)
        detail = levels.copy()
        days = detail["horizon"].map(_DAY_COUNT)
        month_len = detail["issue_date"].dt.days_in_month
        detail["period_start"] = detail["issue_date"]
        detail["period_end"] = detail["issue_date"] + pd.to_timedelta(days.fillna(month_len) - 1, unit="D")
        detail = detail[["level", "unit", "horizon", "issue_date", "period_start", "period_end",
                         "forecast", "actual", "abs_error", "accuracy", "meters"]]
        self._artifacts.save_frame("validation_detail", detail.round(
            {"forecast": 2, "actual": 2, "abs_error": 2, "accuracy": 2}))

    @traced
    def _write_day_ahead(self, panel, preds: pd.DataFrame) -> None:
        """
        #Feature: Accuracy
        #Scenario: Day-ahead forecasts next to gold daily actuals, for charts and audit
        #Complexity: Time O(n) - rows

        Args:
            panel: The panel.
            preds: Every-morning forecasts.

        Raises:
            ValueError: When an argument is None.
        """
        require("AccuracyService._write_day_ahead", "panel", panel)
        day0 = preds[preds["days_ahead"] == 0]
        m, t = day0["meter_idx"].to_numpy(), day0["target_idx"].to_numpy()
        self._artifacts.save_frame("day_ahead_vs_actual", pd.DataFrame({
            "date": panel.dates[t], "meter_label": np.array(panel.meters)[m], "zone": panel.zone[m],
            "day_ahead_forecast_kwh": day0["pred_kwh"].round(2).to_numpy(),
            "actual_kwh": panel.gold[m, t].round(2), "measurable_day": panel.measurable[m, t]}))

    @traced
    def write_validation(self, panel, preds: pd.DataFrame, periods: pd.DataFrame, levels: pd.DataFrame,
                         coefficients: Dict) -> None:
        """
        #Feature: Accuracy
        #Scenario: Save every accuracy table
        #Complexity: Time O(n) - scored periods

        Args:
            panel: The panel.
            preds: Every-morning forecasts.
            periods: Per-meter periods.
            levels: Rows for every level.
            coefficients: Learned weights of each monthly model.

        Raises:
            ValueError: When an argument is None.
        """
        require("AccuracyService.write_validation", "levels", levels)
        self._write_detail(levels)
        summ = summary(levels)
        self._artifacts.save_frame("accuracy_summary", summ)
        for level in _LEVELS:
            for horizon in HORIZON_LABELS.values():
                self._artifacts.save_named_frame(
                    ACCURACY_MONTH_FOLDER, f"accuracy_by_month_{level}_{horizon.replace(' ', '_')}.csv",
                    month_matrix(levels, level, horizon).round(1))
        excluded = periods[~periods["measurable"]]
        self._artifacts.save_frame("periods_not_measurable", excluded[
            ["meter_label", "horizon", "issue_date", "forecast", "actual"]].round({"forecast": 2, "actual": 2}))
        coef = pd.concat({month: frame for month, frame in coefficients.items()}, names=["model_month", "driver"])
        self._artifacts.save_frame("learned_model_weights", coef.round(4), index=True)
        self._write_day_ahead(panel, preds)
        self._artifacts.save_json("accuracy_summary_json", summ.to_dict("records"), indent=2, default=str)

    @traced
    def read_summary(self) -> List[Dict]:
        """
        #Feature: Accuracy
        #Scenario: The saved accuracy summary
        #Complexity: Time O(1) - small file

        Returns:
            One dict per level and horizon.

        Raises:
            Errors.Misc.DataNotReady: When the pipeline has not run yet.
        """
        return self._artifacts.load_json("accuracy_summary_json")

    @traced
    def read_month_matrix(self, level: str, horizon: str) -> Dict:
        """
        #Feature: Accuracy
        #Scenario: The saved unit x month accuracy table
        #Complexity: Time O(n) - table size

        Args:
            level: "meter", "zone" or "facility".
            horizon: "1 day", "7 days", "30 days" or "calendar month".

        Returns:
            Dict with columns and rows.

        Raises:
            Errors.HTTP.BadRequest: When level or horizon is unknown.
            Errors.Misc.DataNotReady: When the pipeline has not run yet.
        """
        require("AccuracyService.read_month_matrix", "level", level)
        require("AccuracyService.read_month_matrix", "horizon", horizon)
        if level not in _LEVELS or horizon not in HORIZON_LABELS.values():
            raise Errors.HTTP.BadRequest(
                f"[AccuracyService.read_month_matrix] Unknown level/horizon. Invariant violated: level in "
                f"{_LEVELS} and horizon in {tuple(HORIZON_LABELS.values())}. Received: {level!r}, {horizon!r}. "
                f"Fix: use one of those.")
        name = f"accuracy_by_month_{level}_{horizon.replace(' ', '_')}.csv"
        table = self._artifacts.load_named_frame(ACCURACY_MONTH_FOLDER, name, index_col=0)
        return {"columns": list(table.columns), "rows": table.reset_index().replace({np.nan: None}).to_dict("records")}
