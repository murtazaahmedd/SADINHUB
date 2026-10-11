"""Daily actual kWh per meter, with the data-quality facts needed to judge it.

Two daily series are produced for every meter-day:

* ``gold_kwh``   - booked the way gold.energy_consumption_validated books it (the
                   energy of each interval lands on the day of its closing
                   reading). Accuracy is measured against this number.
* ``spread_kwh`` - the same energy, but an interval that spans several days (a
                   meter outage) is shared across those days in proportion to
                   time. Only used to train the model, so an outage never
                   teaches it a fake 0 or a fake spike.

A meter-day is *measurable* when readings cover at least 95% of the day and no
outage longer than an hour books energy into it.
"""
from typing import List, Tuple

import numpy as np
import pandas as pd

from constants.data_rules import BOUNDARY_GAP_MINUTES, MIN_DAY_COVERAGE_SHARE, SAUDI_OFFSET_HOURS
from utilities.guard import require, traced

_DAY = pd.Timedelta(days=1)
_SAUDI = pd.Timedelta(hours=SAUDI_OFFSET_HOURS)
_AGG_COLUMNS = ["meter_label", "ts_utc", "kwh", "source"]


@traced
def _split_rows(intervals: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    #Feature: Daily actuals
    #Scenario: Separate intervals that stay inside one day from those spanning several days
    #Complexity: Time O(n) - linear

    Args:
        intervals: Validated intervals with prev_saudi, saudi, date and consumption_kwh.

    Returns:
        (boolean mask of same-day intervals, DataFrame of those rows as meter-day kWh).

    Raises:
        ValueError: When intervals is None.
    """
    require("_split_rows", "intervals", intervals)
    start_day = intervals["prev_saudi"].dt.normalize().to_numpy()
    end_day = intervals["date"].to_numpy()
    same = start_day == end_day
    labels = intervals["meter_label"].to_numpy()
    kwh = intervals["consumption_kwh"].to_numpy()
    return same, pd.DataFrame({"meter_label": labels[same], "date": end_day[same], "spread_kwh": kwh[same]})


@traced
def _share_over_days(label: str, start: np.datetime64, end: np.datetime64, end_day: np.datetime64,
                     start_day: np.datetime64, kwh: float) -> List[tuple]:
    """
    #Feature: Daily actuals
    #Scenario: Share one multi-day interval's kWh across the days it covers, in proportion to time
    #Complexity: Time O(d) - d days spanned

    Args:
        label: Meter label.
        start: Interval start (Saudi time).
        end: Interval end (Saudi time).
        end_day: Day of the closing reading.
        start_day: Day of the opening reading.
        kwh: Energy booked on the interval.

    Returns:
        (meter_label, day, kWh) tuples; empty when the interval has no duration.

    Raises:
        ValueError: When label is None.
    """
    require("_share_over_days", "label", label)
    total_ns = (end - start).astype("timedelta64[ns]").astype(np.int64)
    if total_ns <= 0:
        return []
    rows, day = [], start_day
    while day <= end_day:
        lo = max(start, day)
        hi = min(end, day + np.timedelta64(1, "D"))
        share = (hi - lo).astype("timedelta64[ns]").astype(np.int64) / total_ns
        if share > 0:
            rows.append((label, day, kwh * share))
        day = day + np.timedelta64(1, "D")
    return rows


@traced
def spread_kwh(intervals: pd.DataFrame) -> pd.DataFrame:
    """
    #Feature: Daily actuals
    #Scenario: Daily kWh with outage catch-ups shared over the missing days
    #Complexity: Time O(n + d) - intervals plus days spanned by long intervals

    Args:
        intervals: Validated intervals with prev_saudi, saudi, date and consumption_kwh.

    Returns:
        One row per meter-day with spread_kwh.

    Raises:
        ValueError: When intervals is None.
    """
    require("spread_kwh", "intervals", intervals)
    same, same_rows = _split_rows(intervals)
    start = intervals["prev_saudi"].to_numpy()
    end = intervals["saudi"].to_numpy()
    start_day = intervals["prev_saudi"].dt.normalize().to_numpy()
    end_day = intervals["date"].to_numpy()
    labels = intervals["meter_label"].to_numpy()
    kwh = intervals["consumption_kwh"].to_numpy()
    rows: List[tuple] = []
    for i in np.flatnonzero(~same):
        rows += _share_over_days(labels[i], start[i], end[i], end_day[i], start_day[i], kwh[i])
    pieces = [same_rows] + ([pd.DataFrame(rows, columns=["meter_label", "date", "spread_kwh"])] if rows else [])
    return pd.concat(pieces, ignore_index=True).groupby(["meter_label", "date"], as_index=False)["spread_kwh"].sum()


@traced
def midnight_gaps(intervals: pd.DataFrame) -> pd.DataFrame:
    """
    #Feature: Daily actuals
    #Scenario: Midnights that fall inside an interval longer than the boundary gap
    #Complexity: Time O(g + d) - long intervals plus midnights spanned

    Args:
        intervals: Validated intervals with gap_minutes, prev_saudi and saudi.

    Returns:
        Unique (meter_label, date) pairs whose day starts inside a long gap.

    Raises:
        ValueError: When intervals is None.
    """
    require("midnight_gaps", "intervals", intervals)
    long = intervals[intervals["gap_minutes"] > BOUNDARY_GAP_MINUTES]
    rows = []
    for row in long.itertuples():
        midnight = row.prev_saudi.normalize() + _DAY
        while midnight <= row.saudi:
            if midnight > row.prev_saudi:
                rows.append((row.meter_label, midnight))
            midnight += _DAY
    return pd.DataFrame(rows, columns=["meter_label", "date"]).drop_duplicates()


@traced
def combine_readings(database: pd.DataFrame, excel: pd.DataFrame) -> pd.DataFrame:
    """
    #Feature: Daily actuals
    #Scenario: One sorted, de-duplicated reading table from the database and Excel exports
    #Complexity: Time O(n log n) - sort

    Args:
        database: Database readings (meter_label, ts_utc, kwh, source).
        excel: Excel readings, possibly empty.

    Returns:
        Readings sorted by meter and time, first copy of any duplicate kept.

    Raises:
        ValueError: When an argument is None.
    """
    require("combine_readings", "database", database)
    require("combine_readings", "excel", excel)
    readings = pd.concat([database[_AGG_COLUMNS], excel[_AGG_COLUMNS]], ignore_index=True)
    readings = readings.drop_duplicates(["meter_label", "ts_utc"]).sort_values(["meter_label", "ts_utc"])
    return readings


@traced
def add_saudi_columns(validated: pd.DataFrame) -> pd.DataFrame:
    """
    #Feature: Daily actuals
    #Scenario: Saudi-time closing and opening timestamps of each interval
    #Complexity: Time O(n) - linear

    Args:
        validated: Output of apply_gold_rule.

    Returns:
        The same frame with saudi and prev_saudi columns.

    Raises:
        ValueError: When validated is None.
    """
    require("add_saudi_columns", "validated", validated)
    validated["saudi"] = validated["ts_utc"] + _SAUDI
    validated["prev_saudi"] = validated.groupby("meter_label")["saudi"].shift()
    return validated


@traced
def _quality_aggregates(intervals: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    #Feature: Daily actuals
    #Scenario: Gold kWh, normal-interval coverage, outage catch-up and unresolved decrements per meter-day
    #Complexity: Time O(n) - linear

    Args:
        intervals: Validated intervals excluding each meter's first reading.

    Returns:
        (gold, coverage minutes, outage kWh, unresolved decrement counts).

    Raises:
        ValueError: When intervals is None.
    """
    require("_quality_aggregates", "intervals", intervals)
    gold = (intervals.groupby(["meter_label", "date"])
            .agg(gold_kwh=("consumption_kwh", "sum"), readings=("consumption_kwh", "size")).reset_index())
    normal = intervals[intervals["interval_quality"] == "NORMAL_INTERVAL"]
    coverage = normal.groupby(["meter_label", "date"])["gap_minutes"].sum().rename("normal_minutes")
    outage = intervals[intervals["gap_minutes"] > BOUNDARY_GAP_MINUTES]
    outage_kwh = outage.groupby(["meter_label", "date"])["consumption_kwh"].sum().rename("outage_kwh")
    unresolved = intervals["status"].isin(["UNRESOLVED_DECREMENT", "REGISTER_JUMP"]).groupby(
        [intervals["meter_label"], intervals["date"]]).sum().rename("unresolved_decrements")
    return gold, coverage, outage_kwh, unresolved


@traced
def _full_calendar(validated: pd.DataFrame, last_day: pd.Timestamp) -> pd.DataFrame:
    """
    #Feature: Daily actuals
    #Scenario: A complete calendar per meter from its first valid reading to the data end
    #Complexity: Time O(m*d) - meters times days

    Args:
        validated: Validated intervals.
        last_day: Last Saudi day of the data.

    Returns:
        DataFrame of meter_label and date, one row per meter-day.

    Raises:
        ValueError: When an argument is None.
    """
    require("_full_calendar", "validated", validated)
    require("_full_calendar", "last_day", last_day)
    first_day = validated.groupby("meter_label")["date"].min()
    frames = [pd.DataFrame({"meter_label": m, "date": pd.date_range(d, last_day, freq="D")})
              for m, d in first_day.items()]
    return pd.concat(frames, ignore_index=True)


@traced
def _add_measurable(daily: pd.DataFrame, gaps: pd.DataFrame, last_day: pd.Timestamp) -> pd.DataFrame:
    """
    #Feature: Daily actuals
    #Scenario: Flag which meter-days can be trusted as the energy really used that day
    #Complexity: Time O(n) - linear

    Args:
        daily: Daily table with coverage, outage and unresolved columns.
        gaps: Midnight-gap pairs.
        last_day: Last Saudi day of the data (always incomplete).

    Returns:
        The daily table with coverage_share, measurable_day, complete_day and outage_at_day_start.

    Raises:
        ValueError: When an argument is None.
    """
    require("_add_measurable", "daily", daily)
    require("_add_measurable", "gaps", gaps)
    gaps = gaps.copy()
    gaps["outage_at_day_start"] = True
    daily = daily.merge(gaps, on=["meter_label", "date"], how="left")
    daily["outage_at_day_start"] = daily["outage_at_day_start"].astype("boolean").fillna(False).astype(bool)
    daily["coverage_share"] = (daily["normal_minutes"] / 1440.0).clip(upper=1.0)
    complete = daily["date"] < last_day
    daily["measurable_day"] = (complete & (daily["coverage_share"] >= MIN_DAY_COVERAGE_SHARE)
                               & (daily["outage_kwh"] == 0) & (daily["unresolved_decrements"] == 0))
    daily["complete_day"] = complete
    return daily


@traced
def build_daily_table(readings: pd.DataFrame, validated: pd.DataFrame, zones: pd.Series) -> pd.DataFrame:
    """
    #Feature: Daily actuals
    #Scenario: The daily table every later stage reads
    #Complexity: Time O(n + m*d) - intervals plus the meter-day calendar

    Args:
        readings: Combined readings (meter_label, ts_utc, kwh, source).
        validated: apply_gold_rule output with Saudi columns added.
        zones: Zone number per meter, indexed by meter_label.

    Returns:
        One row per meter-day with gold_kwh, spread_kwh, coverage and quality flags.

    Raises:
        ValueError: When an argument is None, or a meter has no zone.
    """
    require("build_daily_table", "readings", readings)
    require("build_daily_table", "validated", validated)
    require("build_daily_table", "zones", zones)
    source_end = readings.groupby("source")["ts_utc"].max()
    intervals = validated[validated["status"] != "FIRST_VALID_READING"].copy()
    gold, coverage, outage_kwh, unresolved = _quality_aggregates(intervals)
    last_day = (readings["ts_utc"].max() + _SAUDI).normalize()
    daily = (_full_calendar(validated, last_day).merge(gold, on=["meter_label", "date"], how="left")
             .merge(spread_kwh(intervals), on=["meter_label", "date"], how="left")
             .merge(coverage.reset_index(), on=["meter_label", "date"], how="left")
             .merge(outage_kwh.reset_index(), on=["meter_label", "date"], how="left")
             .merge(unresolved.reset_index(), on=["meter_label", "date"], how="left"))
    for col in ("gold_kwh", "spread_kwh", "normal_minutes", "outage_kwh", "readings", "unresolved_decrements"):
        daily[col] = daily[col].fillna(0.0)
    daily = _add_measurable(daily, midnight_gaps(validated.dropna(subset=["prev_saudi"])), last_day)
    database_last_day = (source_end["database"] + _SAUDI).normalize()
    daily["source"] = np.where(daily["date"] <= database_last_day, "database", "excel")
    daily["zone"] = daily["meter_label"].map(zones)
    if daily["zone"].isna().any():
        missing = sorted(daily.loc[daily["zone"].isna(), "meter_label"].unique())
        raise ValueError(f"[build_daily_table] Meters without a zone: {missing}. Invariant violated: every meter "
                         f"has a zone. Received: no zone for these. Fix: check the meter table (gold.zone).")
    daily["zone"] = daily["zone"].astype(int)
    return daily.sort_values(["meter_label", "date"]).reset_index(drop=True)
