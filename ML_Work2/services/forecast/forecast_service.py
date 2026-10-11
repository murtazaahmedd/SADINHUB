"""Training, replaying and forecasting.

A forecast is issued on the morning of day T using actuals up to T-1. The
model in use on T was trained only on data before the first day of T's month,
so nothing that happened on or after T can reach the forecast.
"""
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from constants.forecasting import (LONG_RANGE_DAYS, LONG_RANGE_WARMUP_MONTHS, LONG_RANGE_WINDOWS,
                                   MAX_HORIZON_DAYS, WALK_FORWARD_DAYS)
from errors import Errors
from interfaces.repositories.artifacts.artifact_repository import IArtifactRepository
from interfaces.services.forecast.forecast_service import IForecastService
from models.forecasting.features import Panel, origin_rows
from models.forecasting.forecaster import Forecaster
from utilities.guard import require, traced

_LEVEL_KEYS = {"meters": "forecast_meters", "zones": "forecast_zones", "facility": "forecast_facility"}


class ForecastService(IForecastService):
    """
    #Feature: Forecasting
    #Scenario: Replay history, forecast the year end and check long-range accuracy
    #Complexity: Time O(months * fit) - one model per month

    Orchestrates the Forecaster model and the saved-results store.
    """

    @traced
    def __init__(self, artifacts: IArtifactRepository, make_model: Callable[[], Forecaster] = Forecaster) -> None:
        """
        #Feature: Forecasting
        #Scenario: Constructor injection of the store and the model factory
        #Complexity: Time O(1) - constant

        Args:
            artifacts: Saved-results store.
            make_model: Factory that returns an unfitted model.

        Raises:
            ValueError: When artifacts is None.
        """
        require("ForecastService.__init__", "artifacts", artifacts)
        self._artifacts = artifacts
        self._make_model = make_model

    @traced
    def build_panel(self, daily: pd.DataFrame, weather: pd.DataFrame, end_date) -> Panel:
        """
        #Feature: Forecasting
        #Scenario: The meter x date arrays the model reads
        #Complexity: Time O(m*n) - meters times days

        Args:
            daily: The daily table.
            weather: Observed daily temperature.
            end_date: Last day the panel covers.

        Returns:
            A Panel.

        Raises:
            ValueError: When an argument is None.
        """
        return Panel(daily, weather, end_date)

    @traced
    def _month_starts(self, panel: Panel, first_day, last_day) -> List[int]:
        """
        #Feature: Forecasting
        #Scenario: Day indices at which a new monthly model is trained
        #Complexity: Time O(months) - months

        Args:
            panel: The panel.
            first_day: First issue day.
            last_day: Last issue day.

        Returns:
            Day index of the first issue day and of every month start after it.

        Raises:
            ValueError: When an argument is None.
        """
        require("ForecastService._month_starts", "first_day", first_day)
        require("ForecastService._month_starts", "last_day", last_day)
        days = pd.date_range(pd.Timestamp(first_day), pd.Timestamp(last_day), freq="MS")
        if pd.Timestamp(first_day) not in days:
            days = days.insert(0, pd.Timestamp(first_day))
        return [panel.index_of(d) for d in days]

    @traced
    def walk_forward(self, panel: Panel, first_origin, last_origin, horizons: Optional[int] = None,
                     log: Optional[Callable[[str], None]] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        #Feature: Forecasting
        #Scenario: Replay every morning exactly as it would have run
        #Complexity: Time O(months * fit + days * rows) - monthly models plus daily features

        Args:
            panel: The panel.
            first_origin: First issue day.
            last_origin: Last issue day.
            horizons: Days ahead issued on each morning (default 31).
            log: Progress callback.

        Returns:
            (forecasts for every morning, learned coefficients of each monthly model).

        Raises:
            ValueError: When panel is None.
        """
        require("ForecastService.walk_forward", "panel", panel)
        horizons = horizons or WALK_FORWARD_DAYS
        log = log or print
        first_idx, last_idx = panel.index_of(first_origin), panel.index_of(last_origin)
        starts = self._month_starts(panel, first_origin, last_origin)
        out, coefficients = [], {}
        for i, cutoff in enumerate(starts):
            end = starts[i + 1] if i + 1 < len(starts) else last_idx + 1
            tick = time.time()
            model = self._make_model().fit(panel, cutoff)
            coefficients[panel.dates[cutoff].strftime("%Y-%m")] = model.coefficients()
            for t in range(max(cutoff, first_idx), min(end, last_idx + 1)):
                rows = origin_rows(panel, t, horizons=horizons)
                rows["pred_kwh"] = model.predict_kwh(rows)
                out.append(rows[["meter_idx", "origin_idx", "target_idx", "days_ahead", "pred_kwh", "base_kwh"]])
            log(f"  {panel.dates[cutoff]:%b %Y}: model trained on data to {model.trained_through:%Y-%m-%d} "
                f"({model.training_rows:,} examples, {time.time() - tick:.0f}s)")
        return pd.concat(out, ignore_index=True), coefficients

    @traced
    def forecast_to_year_end(self, panel: Panel, daily: pd.DataFrame,
                             forecast_end) -> Tuple[pd.DataFrame, Forecaster, pd.Timestamp]:
        """
        #Feature: Forecasting
        #Scenario: Train on every complete day and forecast each day to the year end
        #Complexity: Time O(fit + m*h) - one fit plus meters times horizons

        The forecast is issued on the morning after the last complete day, exactly
        like each morning of the validation.

        Args:
            panel: The panel.
            daily: The daily table.
            forecast_end: Last day to forecast.

        Returns:
            (daily forecast per meter, fitted model, issue day).

        Raises:
            ValueError: When an argument is None.
            Errors.Misc.HorizonExceeded: When the forecast needs more days than the model was trained for.
        """
        require("ForecastService.forecast_to_year_end", "panel", panel)
        require("ForecastService.forecast_to_year_end", "forecast_end", forecast_end)
        issue = daily.loc[daily["complete_day"], "date"].max() + pd.Timedelta(days=1)
        t = panel.index_of(issue)
        horizons = panel.index_of(forecast_end) - t + 1
        if horizons > MAX_HORIZON_DAYS:
            raise Errors.Misc.HorizonExceeded(
                f"[ForecastService.forecast_to_year_end] The forecast needs {horizons} days ahead but the model is "
                f"trained for {MAX_HORIZON_DAYS}. Invariant violated: horizon <= MAX_HORIZON_DAYS. Received: issue "
                f"{issue:%Y-%m-%d}, end {forecast_end}. Fix: load newer data or raise "
                f"constants.forecasting.MAX_HORIZON_DAYS.")
        model = self._make_model().fit(panel, t)
        rows = origin_rows(panel, t, horizons=horizons)
        rows["forecast_kwh"] = model.predict_kwh(rows)
        out = pd.DataFrame({
            "date": panel.dates[rows["target_idx"].to_numpy()],
            "meter_label": np.array(panel.meters)[rows["meter_idx"].to_numpy()],
            "zone": panel.zone[rows["meter_idx"].to_numpy()],
            "days_ahead": rows["days_ahead"].to_numpy() + 1,
            "forecast_kwh": rows["forecast_kwh"].round(2),
        })
        return out, model, issue

    @traced
    def write_forecast(self, daily: pd.DataFrame, future: pd.DataFrame, model: Forecaster, issue: pd.Timestamp,
                       forecast_end) -> None:
        """
        #Feature: Forecasting
        #Scenario: Save the year-end forecast by meter, zone and facility, plus the monthly outlook
        #Complexity: Time O(n) - rows

        Args:
            daily: The daily table.
            future: Year-end forecast per meter.
            model: The fitted model.
            issue: Issue day.
            forecast_end: Last forecast day.

        Raises:
            ValueError: When an argument is None.
        """
        require("ForecastService.write_forecast", "future", future)
        require("ForecastService.write_forecast", "model", model)
        year = pd.Timestamp(forecast_end).year
        self._artifacts.save_frame("forecast_meters", future)
        self._artifacts.save_frame("forecast_zones", future.groupby(["date", "zone"], as_index=False)["forecast_kwh"].sum())
        self._artifacts.save_frame("forecast_facility", future.groupby("date", as_index=False)["forecast_kwh"].sum())
        actual = daily[daily["complete_day"] & (daily["date"].dt.year == year)]
        act_m = actual.groupby([actual["date"].dt.strftime("%Y-%m"), "meter_label"])["gold_kwh"].sum()
        fut_m = future.groupby([future["date"].dt.strftime("%Y-%m"), "meter_label"])["forecast_kwh"].sum()
        monthly = pd.concat({"actual_kwh": act_m, "forecast_kwh": fut_m}, axis=1).fillna(0.0)
        monthly["outlook_kwh"] = monthly["actual_kwh"] + monthly["forecast_kwh"]
        monthly.index.names = ["month", "meter_label"]
        self._artifacts.save_frame("monthly_outlook", monthly.round(1), index=True)
        self._artifacts.save_frame("forecast_weights", model.coefficients().round(4), index=True)
        self._artifacts.save_json("forecast_info", {
            "forecast_issued_for_morning_of": f"{issue:%Y-%m-%d}",
            "uses_actuals_through": f"{issue - pd.Timedelta(days=1):%Y-%m-%d}",
            "forecast_until": f"{pd.Timestamp(forecast_end):%Y-%m-%d}",
            "model_trained_through": f"{model.trained_through:%Y-%m-%d}",
            "training_examples": model.training_rows}, indent=2)

    @traced
    def _actual_matrix(self, panel: Panel, daily: pd.DataFrame) -> np.ndarray:
        """
        #Feature: Forecasting
        #Scenario: Trusted daily kWh with outage catch-ups spread over the days they belong to
        #Complexity: Time O(m*n) - meters times days

        Args:
            panel: The panel.
            daily: The daily table.

        Returns:
            Meter x day array of spread_kwh on complete days.

        Raises:
            ValueError: When an argument is None.
        """
        require("ForecastService._actual_matrix", "daily", daily)
        return (daily[daily["complete_day"]].pivot_table(index="meter_label", columns="date", values="spread_kwh")
                .reindex(index=panel.meters, columns=panel.dates).to_numpy())

    @traced
    def _replay_month(self, panel: Panel, actual: np.ndarray, start: pd.Timestamp, last_actual: int) -> List[dict]:
        """
        #Feature: Forecasting
        #Scenario: One forecast issued on the 1st of a month, scored in 30-day blocks
        #Complexity: Time O(fit + m*h) - one fit plus meters times horizons

        Args:
            panel: The panel.
            actual: Output of _actual_matrix.
            start: The issue day (1st of a month).
            last_actual: Day index of the last complete day.

        Returns:
            One dict per 30-day window with forecast and actual facility kWh.

        Raises:
            ValueError: When panel is None.
        """
        require("ForecastService._replay_month", "panel", panel)
        t = panel.index_of(start)
        model = self._make_model().fit(panel, t)
        pred = origin_rows(panel, t, horizons=LONG_RANGE_DAYS)
        pred["kwh"] = model.predict_kwh(pred)
        rows = []
        for lo, label in LONG_RANGE_WINDOWS:
            if t + lo + 30 > last_actual + 1:
                continue
            forecast = pred.loc[pred["days_ahead"].between(lo, lo + 29), "kwh"].sum()
            real = np.nansum(actual[:, t + lo:t + lo + 30])
            rows.append({"issued": f"{start:%Y-%m-%d}", "window": label,
                         "forecast_kwh": round(forecast, 1), "actual_kwh": round(real, 1)})
        return rows

    @traced
    def long_range_check(self, panel: Panel, daily: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        #Feature: Forecasting
        #Scenario: How accurate were facility forecasts 1-30, 31-60 and 61-90 days ahead?
        #Complexity: Time O(months * fit) - one model per month

        Replays a forecast on the 1st of every month (model trained only on data
        before that day) and compares each 30-day block with the actual facility kWh.

        Args:
            panel: The panel.
            daily: The daily table.

        Returns:
            (detail per issue month and window, overall accuracy per window).

        Raises:
            ValueError: When an argument is None.
        """
        require("ForecastService.long_range_check", "panel", panel)
        actual = self._actual_matrix(panel, daily)
        last_actual = panel.index_of(daily.loc[daily["complete_day"], "date"].max())
        first = (panel.dates[0].to_period("M") + LONG_RANGE_WARMUP_MONTHS).to_timestamp()
        rows: List[dict] = []
        for start in pd.date_range(first, panel.dates[last_actual], freq="MS"):
            if panel.index_of(start) + 30 > last_actual + 1:
                break
            rows += self._replay_month(panel, actual, start, last_actual)
        out = pd.DataFrame(rows)
        out["accuracy"] = (100 * (1 - (out["forecast_kwh"] - out["actual_kwh"]).abs() / out["actual_kwh"])).round(1)
        overall = out.groupby("window").apply(
            lambda g: round(100 * (1 - (g["forecast_kwh"] - g["actual_kwh"]).abs().sum() / g["actual_kwh"].sum()), 1),
            include_groups=False)
        return out, overall.to_dict()

    @traced
    def read_forecast(self, level: str, start: Optional[str], end: Optional[str], page: int, limit: int) -> Dict:
        """
        #Feature: Forecasting
        #Scenario: Page through the saved forecast
        #Complexity: Time O(n) - rows

        Args:
            level: "meters", "zones" or "facility".
            start: Optional first date (YYYY-MM-DD).
            end: Optional last date (YYYY-MM-DD).
            page: 1-based page.
            limit: Rows per page.

        Returns:
            Dict with total, page, limit and rows.

        Raises:
            Errors.HTTP.BadRequest: When level is unknown.
            Errors.Misc.DataNotReady: When the pipeline has not run yet.
        """
        require("ForecastService.read_forecast", "level", level)
        if level not in _LEVEL_KEYS:
            raise Errors.HTTP.BadRequest(f"[ForecastService.read_forecast] Unknown level {level!r}. Invariant "
                                         f"violated: level in {sorted(_LEVEL_KEYS)}. Received: {level!r}. "
                                         f"Fix: use one of those.")
        frame = self._artifacts.load_frame(_LEVEL_KEYS[level], parse_dates=["date"])
        if start:
            frame = frame[frame["date"] >= pd.Timestamp(start)]
        if end:
            frame = frame[frame["date"] <= pd.Timestamp(end)]
        window = frame.iloc[(page - 1) * limit:page * limit].assign(date=lambda d: d["date"].dt.strftime("%Y-%m-%d"))
        return {"total": int(len(frame)), "page": page, "limit": limit, "rows": window.to_dict("records")}
