"""Load the two saved inputs (read-only) and build one row per meter per day.

Inputs, produced by the forecasting pipeline and never modified here:
  * outputs/data/daily_actuals.parquet                  - gold-booked daily kWh + quality fields
  * outputs/validation/every_morning_forecast_2026.parquet - every forecast made each morning

Conventions (same as forecasting/features.py):
  * meter_idx  = position in the sorted list of meter labels
  * origin_idx / target_idx = days since the earliest date in the daily table
  * days_ahead = 0 is the forecast made on the morning of that same day,
    using actual readings up to the day before.
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DAILY_PATH = ROOT / "outputs" / "data" / "daily_actuals.parquet"
FORECAST_PATH = ROOT / "outputs" / "validation" / "every_morning_forecast_2026.parquet"
INTERVALS_PATH = ROOT / "outputs" / "data" / "validated_intervals.parquet"
LONG_GAP_HOURS_FROM_MINUTES = 60.0   # a reading gap longer than this counts as missing time


def fingerprint(path):
    """SHA-256 of a file, so outputs can say exactly which inputs they came from."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_inputs():
    daily = pd.read_parquet(DAILY_PATH)
    forecasts = pd.read_parquet(FORECAST_PATH)
    return daily, forecasts


def _check_mapping(daily, day0, meters, start):
    """Prove the index -> meter/date mapping by recomputing the forecast's own base level.

    Each saved forecast row carries `base_kwh`: the mean of the meter's
    trusted (measurable) daily kWh over the 7 days before the forecast
    morning, when at least 4 such days exist. Recomputing it from the daily
    table with our mapping must give the same number; a wrong meter or a
    shifted date would not.
    """
    trusted = daily[daily["measurable_day"]].pivot_table(index="meter_label", columns="date", values="spread_kwh")
    trusted = trusted.reindex(index=meters, columns=pd.date_range(start, daily["date"].max()))
    values = trusted.to_numpy()
    checked = mismatched = 0
    for row in day0.itertuples():
        window = values[row.meter_idx, row.origin_idx - 7:row.origin_idx]
        window = window[~np.isnan(window)]
        if len(window) >= 4:
            checked += 1
            if abs(window.mean() - row.base_kwh) > 1e-6 * max(1.0, abs(row.base_kwh)):
                mismatched += 1
    if checked < 0.8 * len(day0) or mismatched:
        raise RuntimeError(f"Forecast-to-meter/date mapping check failed: {mismatched} of {checked} rows differ.")
    return checked


def missing_hours_per_day():
    """Hours per meter-day that fall inside reading gaps longer than an hour.

    A meter that reports every 15-60 minutes instead of every few minutes has
    lower "normal-interval coverage" but no missing time; only real gaps count
    here. A gap that spans midnight is split across the days it covers.
    """
    iv = pd.read_parquet(INTERVALS_PATH, columns=["meter_label", "prev_saudi", "saudi", "gap_minutes"])
    iv = iv.dropna(subset=["prev_saudi"])
    iv = iv[(iv["gap_minutes"] > LONG_GAP_HOURS_FROM_MINUTES) & (iv["saudi"] >= "2026-01-01")]
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


def meter_days():
    """One row per meter per 2026 forecast day: expected (morning forecast), actual and quality.

    `actual_kwh` is the gold-booked kWh and is only filled on days whose
    readings are complete and trustworthy (`eligible`); every other day is a
    data-quality day and its booked kWh is never read as consumption.
    """
    daily, forecasts = load_inputs()
    meters = sorted(daily["meter_label"].unique())
    start = daily["date"].min()
    day0 = forecasts[forecasts["days_ahead"] == 0].copy()
    if not (day0["target_idx"] == day0["origin_idx"]).all():
        raise RuntimeError("Same-day forecasts must target the morning they were made.")
    if day0.duplicated(["meter_idx", "target_idx"]).any():
        raise RuntimeError("More than one same-day forecast for a meter-day.")
    checked = _check_mapping(daily, day0, meters, start)

    day0["meter_label"] = np.array(meters)[day0["meter_idx"].to_numpy()]
    day0["date"] = start + pd.to_timedelta(day0["target_idx"], unit="D")
    frame = day0[["meter_label", "date", "pred_kwh"]].rename(columns={"pred_kwh": "expected_kwh"})
    cols = ["meter_label", "date", "zone", "gold_kwh", "readings", "coverage_share", "outage_kwh",
            "unresolved_decrements", "measurable_day", "complete_day", "source", "outage_at_day_start"]
    frame = frame.merge(daily[cols], on=["meter_label", "date"], how="left", validate="one_to_one")
    if frame["complete_day"].isna().any():
        raise RuntimeError("Some forecast days have no row in the daily table.")
    frame = frame.merge(missing_hours_per_day(), on=["meter_label", "date"], how="left")
    frame["gap_hours"] = frame["gap_hours"].fillna(0.0)
    frame.loc[frame["readings"] == 0, "gap_hours"] = 24.0   # nothing at all was reported that day
    frame = frame[frame["complete_day"].astype(bool)].copy()
    frame["eligible"] = frame["measurable_day"].astype(bool)
    frame["actual_kwh"] = np.where(frame["eligible"], frame["gold_kwh"], np.nan)
    frame = frame.sort_values(["meter_label", "date"]).reset_index(drop=True)
    frame.attrs["mapping_rows_checked"] = checked
    frame.attrs["inputs"] = {"daily_actuals": fingerprint(DAILY_PATH), "every_morning_forecast": fingerprint(FORECAST_PATH)}
    # Rebuilt together with daily_actuals by the forecasting pipeline; recorded for traceability.
    frame.attrs["other_inputs"] = {"validated_intervals": fingerprint(INTERVALS_PATH)}
    return frame
