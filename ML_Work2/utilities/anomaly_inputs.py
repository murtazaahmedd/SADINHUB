"""Build one row per meter per forecast day from the saved forecasting outputs (never modified here).

Conventions (same as the forecasting features):
  * meter_idx  = position in the sorted list of meter labels
  * origin_idx / target_idx = days since the earliest date in the daily table
  * days_ahead = 0 is the forecast made on the morning of that same day, using
    actual readings up to the day before.
"""
from typing import List

import numpy as np
import pandas as pd

from constants.anomaly import GAP_HOURS_MINUTES
from errors import Errors
from utilities.guard import require, traced


@traced
def check_mapping(daily: pd.DataFrame, day0: pd.DataFrame, meters: List[str], start: pd.Timestamp) -> int:
    """
    #Feature: Anomaly inputs
    #Scenario: Prove the index -> meter/date mapping by recomputing the forecast's own base level
    #Complexity: Time O(rows) - same-day forecasts

    Each saved forecast row carries base_kwh: the mean of the meter's trusted daily kWh
    over the 7 days before the forecast morning, when at least 4 such days exist.
    A wrong meter or a shifted date would not reproduce it.

    Args:
        daily: The daily table.
        day0: Same-day forecasts.
        meters: Sorted meter labels.
        start: First date of the daily table.

    Returns:
        Number of rows checked.

    Raises:
        ValueError: When an argument is None.
        Errors.HTTP.InternalServerError: When the mapping does not reproduce the saved base level.
    """
    require("check_mapping", "daily", daily)
    require("check_mapping", "day0", day0)
    trusted = daily[daily["measurable_day"]].pivot_table(index="meter_label", columns="date", values="spread_kwh")
    values = trusted.reindex(index=meters, columns=pd.date_range(start, daily["date"].max())).to_numpy()
    checked = mismatched = 0
    for row in day0.itertuples():
        window = values[row.meter_idx, row.origin_idx - 7:row.origin_idx]
        window = window[~np.isnan(window)]
        if len(window) >= 4:
            checked += 1
            if abs(window.mean() - row.base_kwh) > 1e-6 * max(1.0, abs(row.base_kwh)):
                mismatched += 1
    if checked < 0.8 * len(day0) or mismatched:
        raise Errors.HTTP.InternalServerError(
            f"[check_mapping] Forecast-to-meter/date mapping check failed: {mismatched} of {checked} rows differ. "
            f"Invariant violated: saved base level reproduces from the daily table. Received: {mismatched} "
            f"mismatches. Fix: rerun the forecasting pipeline so both files come from one run.")
    return checked


@traced
def missing_hours_per_day(intervals: pd.DataFrame, since: pd.Timestamp) -> pd.DataFrame:
    """
    #Feature: Anomaly inputs
    #Scenario: Hours per meter-day that fall inside reading gaps longer than an hour
    #Complexity: Time O(g*d) - long gaps times days spanned

    A meter that reports every 15-60 minutes has lower "normal-interval coverage"
    but no missing time; only real gaps count. A gap spanning midnight is split
    across the days it covers.

    Args:
        intervals: Validated intervals (meter_label, prev_saudi, saudi, gap_minutes).
        since: Only gaps closing on or after this day count.

    Returns:
        Columns meter_label, date and gap_hours.

    Raises:
        ValueError: When an argument is None.
    """
    require("missing_hours_per_day", "intervals", intervals)
    require("missing_hours_per_day", "since", since)
    iv = intervals.dropna(subset=["prev_saudi"])
    iv = iv[(iv["gap_minutes"] > GAP_HOURS_MINUTES) & (iv["saudi"] >= since)]
    one = pd.Timedelta(days=1)
    rows = []
    for r in iv.itertuples():
        day = r.prev_saudi.normalize()
        while day < r.saudi:
            overlap = (min(r.saudi, day + one) - max(r.prev_saudi, day)).total_seconds() / 3600.0
            if overlap > 0:
                rows.append((r.meter_label, day, overlap))
            day += one
    out = pd.DataFrame(rows, columns=["meter_label", "date", "gap_hours"])
    return out.groupby(["meter_label", "date"], as_index=False)["gap_hours"].sum()


@traced
def _attach_daily(frame: pd.DataFrame, daily: pd.DataFrame, intervals: pd.DataFrame, start: pd.Timestamp,
                  first_target: int) -> pd.DataFrame:
    """
    #Feature: Anomaly inputs
    #Scenario: Join each forecast day with its quality facts and decide which days can be judged
    #Complexity: Time O(n + g*d) - rows plus gap days

    Args:
        frame: Same-day forecasts with meter_label, date and expected_kwh.
        daily: The daily table.
        intervals: Validated intervals.
        start: First date of the daily table.
        first_target: Day index of the first forecast day.

    Returns:
        The frame with quality columns, gap_hours, eligible and actual_kwh (complete days only).

    Raises:
        ValueError: When an argument is None.
        Errors.HTTP.InternalServerError: When a forecast day has no row in the daily table.
    """
    require("_attach_daily", "frame", frame)
    require("_attach_daily", "daily", daily)
    cols = ["meter_label", "date", "zone", "gold_kwh", "readings", "coverage_share", "outage_kwh",
            "unresolved_decrements", "measurable_day", "complete_day", "source", "outage_at_day_start"]
    frame = frame.merge(daily[cols], on=["meter_label", "date"], how="left", validate="one_to_one")
    if frame["complete_day"].isna().any():
        raise Errors.HTTP.InternalServerError("[build_meter_days] Some forecast days have no row in the daily table. "
                                              "Invariant violated: every forecast day is in the daily table. "
                                              "Fix: rerun the forecasting pipeline.")
    frame = frame.merge(missing_hours_per_day(intervals, start + pd.to_timedelta(first_target, "D")),
                        on=["meter_label", "date"], how="left")
    frame["gap_hours"] = frame["gap_hours"].fillna(0.0)
    frame.loc[frame["readings"] == 0, "gap_hours"] = 24.0   # nothing at all was reported that day
    frame = frame[frame["complete_day"].astype(bool)].copy()
    frame["eligible"] = frame["measurable_day"].astype(bool)
    frame["actual_kwh"] = np.where(frame["eligible"], frame["gold_kwh"], np.nan)
    return frame


@traced
def build_meter_days(daily: pd.DataFrame, forecasts: pd.DataFrame, intervals: pd.DataFrame) -> pd.DataFrame:
    """
    #Feature: Anomaly inputs
    #Scenario: One row per meter per forecast day: expected, actual and data quality
    #Complexity: Time O(n + g*d) - rows plus gap days

    actual_kwh is the gold-booked kWh and is only filled on days whose readings
    are complete and trustworthy (eligible); every other day is a data-quality day
    and its booked kWh is never read as consumption.

    Args:
        daily: The daily table.
        forecasts: Every-morning forecasts.
        intervals: Validated intervals.

    Returns:
        The meter-day frame (attrs carries mapping_rows_checked).

    Raises:
        ValueError: When an argument is None.
        Errors.HTTP.InternalServerError: When the saved files disagree with each other.
    """
    require("build_meter_days", "daily", daily)
    require("build_meter_days", "forecasts", forecasts)
    meters = sorted(daily["meter_label"].unique())
    start = daily["date"].min()
    day0 = forecasts[forecasts["days_ahead"] == 0].copy()
    if not (day0["target_idx"] == day0["origin_idx"]).all() or day0.duplicated(["meter_idx", "target_idx"]).any():
        raise Errors.HTTP.InternalServerError(
            "[build_meter_days] Same-day forecasts must target the morning they were made, once per meter-day. "
            "Invariant violated: one same-day forecast per meter-day. Received: a mismatch or duplicate. "
            "Fix: rerun the forecasting pipeline.")
    checked = check_mapping(daily, day0, meters, start)
    day0["meter_label"] = np.array(meters)[day0["meter_idx"].to_numpy()]
    day0["date"] = start + pd.to_timedelta(day0["target_idx"], unit="D")
    frame = day0[["meter_label", "date", "pred_kwh"]].rename(columns={"pred_kwh": "expected_kwh"})
    frame = _attach_daily(frame, daily, intervals, start, int(day0["target_idx"].min()))
    frame = frame.sort_values(["meter_label", "date"]).reset_index(drop=True)
    frame.attrs["mapping_rows_checked"] = checked
    return frame
