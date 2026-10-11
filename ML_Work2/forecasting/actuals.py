"""Daily actual kWh per meter, with the data-quality facts needed to judge it.

Two daily series are produced for every meter-day:

* ``gold_kwh``   - booked exactly the way gold.energy_consumption_validated books
                   it (energy of each interval lands on the day of its closing
                   reading). This is the number accuracy is measured against.
* ``spread_kwh`` - the same energy, but an interval that spans several days
                   (a meter outage) is shared across those days in proportion to
                   time. Totals are identical; it is only used to train the
                   model, so an outage never teaches it a fake 0 or a fake spike.

A meter-day is *measurable* when readings cover at least 95% of the day and no
outage longer than an hour books energy into it, because only then is the gold
value the energy actually used that day.
"""
import numpy as np
import pandas as pd

from . import config
from .readings import apply_gold_rule, load_database_readings, load_excel_readings

DAILY_PATH = config.DATA_DIR / "daily_actuals.parquet"
INTERVALS_PATH = config.DATA_DIR / "validated_intervals.parquet"

_DAY = pd.Timedelta(days=1)
_SAUDI = pd.Timedelta(hours=config.SAUDI_OFFSET_HOURS)


def _spread(intervals):
    """Share each interval's kWh across the Saudi days it spans."""
    start = intervals["prev_saudi"].to_numpy()
    end = intervals["saudi"].to_numpy()
    kwh = intervals["consumption_kwh"].to_numpy()
    start_day = intervals["prev_saudi"].dt.normalize().to_numpy()
    end_day = intervals["date"].to_numpy()
    same = start_day == end_day

    pieces = [pd.DataFrame({"meter_label": intervals["meter_label"].to_numpy()[same],
                            "date": end_day[same], "spread_kwh": kwh[same]})]
    rows = []
    labels = intervals["meter_label"].to_numpy()
    for i in np.flatnonzero(~same):
        total_ns = (end[i] - start[i]).astype("timedelta64[ns]").astype(np.int64)
        if total_ns <= 0:
            continue
        day = start_day[i]
        while day <= end_day[i]:
            lo = max(start[i], day)
            hi = min(end[i], day + np.timedelta64(1, "D"))
            share = (hi - lo).astype("timedelta64[ns]").astype(np.int64) / total_ns
            if share > 0:
                rows.append((labels[i], day, kwh[i] * share))
            day = day + np.timedelta64(1, "D")
    if rows:
        pieces.append(pd.DataFrame(rows, columns=["meter_label", "date", "spread_kwh"]))
    out = pd.concat(pieces, ignore_index=True)
    return out.groupby(["meter_label", "date"], as_index=False)["spread_kwh"].sum()


def _midnight_gaps(intervals):
    """Midnights that fall inside an interval longer than BOUNDARY_GAP_MINUTES."""
    long = intervals[intervals["gap_minutes"] > config.BOUNDARY_GAP_MINUTES]
    rows = []
    for row in long.itertuples():
        midnight = row.prev_saudi.normalize() + _DAY
        while midnight <= row.saudi:
            if midnight > row.prev_saudi:
                rows.append((row.meter_label, midnight))
            midnight += _DAY
    return pd.DataFrame(rows, columns=["meter_label", "date"]).drop_duplicates()


def build_daily_actuals(refresh=False, refresh_sources=False):
    """Daily table; ``refresh`` rebuilds it, ``refresh_sources`` also re-reads DB and Excel."""
    if DAILY_PATH.exists() and not (refresh or refresh_sources):
        return pd.read_parquet(DAILY_PATH)

    database = load_database_readings(refresh=refresh_sources)
    excel = load_excel_readings(database, refresh=refresh_sources)
    readings = pd.concat([database[["meter_label", "ts_utc", "kwh", "source"]],
                          excel[["meter_label", "ts_utc", "kwh", "source"]]], ignore_index=True)
    readings = readings.drop_duplicates(["meter_label", "ts_utc"]).sort_values(["meter_label", "ts_utc"])
    source_end = readings.groupby("source")["ts_utc"].max()

    validated = apply_gold_rule(readings)
    validated["saudi"] = validated["ts_utc"] + _SAUDI
    validated["prev_saudi"] = validated.groupby("meter_label")["saudi"].shift()
    validated.to_parquet(INTERVALS_PATH, index=False)

    intervals = validated[validated["status"] != "FIRST_VALID_READING"].copy()
    gold = (intervals.groupby(["meter_label", "date"])
            .agg(gold_kwh=("consumption_kwh", "sum"), readings=("consumption_kwh", "size"))
            .reset_index())
    normal = intervals[intervals["interval_quality"] == "NORMAL_INTERVAL"]
    coverage = normal.groupby(["meter_label", "date"])["gap_minutes"].sum().rename("normal_minutes")
    outage = intervals[intervals["gap_minutes"] > config.BOUNDARY_GAP_MINUTES]
    outage_kwh = outage.groupby(["meter_label", "date"])["consumption_kwh"].sum().rename("outage_kwh")
    unresolved = intervals["status"].isin(["UNRESOLVED_DECREMENT", "REGISTER_JUMP"]).groupby(
        [intervals["meter_label"], intervals["date"]]).sum().rename("unresolved_decrements")

    # A complete calendar per meter, from its first valid reading to the data end.
    last_day = (readings["ts_utc"].max() + _SAUDI).normalize()
    first_day = validated.groupby("meter_label")["date"].min()
    frames = [pd.DataFrame({"meter_label": m, "date": pd.date_range(d, last_day, freq="D")})
              for m, d in first_day.items()]
    daily = pd.concat(frames, ignore_index=True)
    daily = (daily.merge(gold, on=["meter_label", "date"], how="left")
             .merge(_spread(intervals), on=["meter_label", "date"], how="left")
             .merge(coverage.reset_index(), on=["meter_label", "date"], how="left")
             .merge(outage_kwh.reset_index(), on=["meter_label", "date"], how="left")
             .merge(unresolved.reset_index(), on=["meter_label", "date"], how="left"))
    for col in ("gold_kwh", "spread_kwh", "normal_minutes", "outage_kwh", "readings", "unresolved_decrements"):
        daily[col] = daily[col].fillna(0.0)

    gaps = _midnight_gaps(validated.dropna(subset=["prev_saudi"]))
    gaps["outage_at_day_start"] = True
    daily = daily.merge(gaps, on=["meter_label", "date"], how="left")
    daily["outage_at_day_start"] = daily["outage_at_day_start"].astype("boolean").fillna(False).astype(bool)

    daily["coverage_share"] = (daily["normal_minutes"] / 1440.0).clip(upper=1.0)
    # The final day of the data is partial, so it is never measurable.
    complete = daily["date"] < last_day
    daily["measurable_day"] = (complete
                               & (daily["coverage_share"] >= config.MIN_DAY_COVERAGE_SHARE)
                               & (daily["outage_kwh"] == 0)
                               & (daily["unresolved_decrements"] == 0))
    daily["complete_day"] = complete
    database_last_day = (source_end["database"] + _SAUDI).normalize()
    daily["source"] = np.where(daily["date"] <= database_last_day, "database", "excel")
    daily["zone"] = daily["meter_label"].str.extract(r"^ZN\.(\d+)-")[0].astype(int)
    daily = daily.sort_values(["meter_label", "date"]).reset_index(drop=True)
    daily.to_parquet(DAILY_PATH, index=False)
    return daily
