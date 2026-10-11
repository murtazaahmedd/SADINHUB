"""The gold-layer consumption rule, reproduced from raw cumulative kWh readings.

It reproduces silver.fact_energy_consumption_validated: startup exclusions,
isolated low/high glitches dropped, consumption equal to the cumulative
register minus the previous valid reading, and LONG_GAP for any interval of
15 minutes or more.
"""
import numpy as np
import pandas as pd

from constants.data_rules import (LONG_GAP_MINUTES, MAX_PLAUSIBLE_METER_KW, SAUDI_OFFSET_HOURS,
                                  STARTUP_EXCLUDED_BEFORE_UTC)
from utilities.guard import require, traced


@traced
def _mark_startup(status: np.ndarray, ts: np.ndarray, label: str) -> None:
    """
    #Feature: Gold consumption rule
    #Scenario: Installation readings before a meter's approved baseline are ignored
    #Complexity: Time O(n) - linear

    Args:
        status: Per-reading status array, edited in place.
        ts: Reading timestamps (datetime64).
        label: Meter label.

    Returns:
        None. ``status`` is edited in place.

    Raises:
        ValueError: When an argument is None.
    """
    require("_mark_startup", "status", status)
    require("_mark_startup", "ts", ts)
    cutoff = STARTUP_EXCLUDED_BEFORE_UTC.get(label)
    if cutoff is not None:
        status[ts < np.datetime64(cutoff)] = "STARTUP_EXCLUDED"


@traced
def _mark_isolated(status: np.ndarray, c: np.ndarray) -> None:
    """
    #Feature: Gold consumption rule
    #Scenario: A single reading that dips or spikes and then recovers is ignored
    #Complexity: Time O(n) - linear

    Args:
        status: Per-reading status array, edited in place.
        c: Cumulative kWh readings.

    Returns:
        None. ``status`` is edited in place.

    Raises:
        ValueError: When an argument is None.
    """
    require("_mark_isolated", "status", status)
    require("_mark_isolated", "c", c)
    prev = np.r_[np.nan, c[:-1]]
    nxt = np.r_[c[1:], np.nan]
    live = status == "KEEP"
    low_bad = live & (c < prev) & (nxt >= prev)
    high_bad = live & (c > nxt) & (nxt >= prev)
    status[low_bad] = "ISOLATED_LOW_BAD"
    status[high_bad] = "ISOLATED_HIGH_BAD"


@traced
def _flag_intervals(kept: pd.DataFrame) -> pd.DataFrame:
    """
    #Feature: Gold consumption rule
    #Scenario: Differences, gaps, unresolved decrements and impossible jumps
    #Complexity: Time O(n) - linear

    Args:
        kept: Kept readings with columns kwh and ts_utc.

    Returns:
        The same frame with consumption_kwh, gap_minutes, status and interval_quality.

    Raises:
        ValueError: When kept is None.
    """
    require("_flag_intervals", "kept", kept)
    kept["consumption_kwh"] = kept["kwh"].diff()
    kept["gap_minutes"] = kept["ts_utc"].diff().dt.total_seconds() / 60
    kept["status"] = "KEEP"
    first = kept.index[0]
    kept.loc[first, "status"] = "FIRST_VALID_READING"
    unresolved = kept["consumption_kwh"] < 0
    kept.loc[unresolved, "status"] = "UNRESOLVED_DECREMENT"
    kept.loc[unresolved, "consumption_kwh"] = 0.0
    implied_kw = kept["consumption_kwh"] / (kept["gap_minutes"] / 60.0)
    jump = (kept["gap_minutes"] < LONG_GAP_MINUTES) & (implied_kw > MAX_PLAUSIBLE_METER_KW)
    kept.loc[jump, "status"] = "REGISTER_JUMP"
    kept.loc[jump, "consumption_kwh"] = 0.0
    kept["interval_quality"] = np.where(kept["gap_minutes"] >= LONG_GAP_MINUTES, "LONG_GAP", "NORMAL_INTERVAL")
    kept.loc[first, "interval_quality"] = "FIRST_VALID_READING"
    return kept


@traced
def _validate_meter(label: str, frame: pd.DataFrame) -> pd.DataFrame:
    """
    #Feature: Gold consumption rule
    #Scenario: Validate one meter's readings
    #Complexity: Time O(n) - linear in the meter's readings

    Args:
        label: Meter label.
        frame: That meter's readings sorted by ts_utc, with columns meter_label, ts_utc, kwh.

    Returns:
        One row per kept reading with consumption and quality columns.

    Raises:
        ValueError: When an argument is None.
    """
    require("_validate_meter", "label", label)
    require("_validate_meter", "frame", frame)
    frame = frame.reset_index(drop=True)
    ts = frame["ts_utc"].to_numpy()
    c = frame["kwh"].to_numpy(dtype=float)
    status = np.full(len(c), "KEEP", dtype=object)
    _mark_startup(status, ts, label)
    _mark_isolated(status, c)
    keep = status == "KEEP"
    kept = frame.loc[keep, ["meter_label", "ts_utc", "kwh"]].copy()
    kept = _flag_intervals(kept)
    kept["dropped_readings"] = int((~keep).sum())
    return kept


@traced
def apply_gold_rule(readings: pd.DataFrame) -> pd.DataFrame:
    """
    #Feature: Gold consumption rule
    #Scenario: Validated per-reading consumption for every meter
    #Complexity: Time O(n log n) - sort plus linear passes

    Args:
        readings: Columns meter_label, ts_utc, kwh.

    Returns:
        One row per kept reading with meter_label, ts_utc, kwh, consumption_kwh,
        gap_minutes, status, interval_quality, dropped_readings and Saudi date.

    Raises:
        ValueError: When readings is None.
    """
    require("apply_gold_rule", "readings", readings)
    results = [_validate_meter(label, frame)
               for label, frame in readings.sort_values(["meter_label", "ts_utc"]).groupby("meter_label", sort=False)]
    out = pd.concat(results, ignore_index=True)
    out["date"] = (out["ts_utc"] + pd.Timedelta(hours=SAUDI_OFFSET_HOURS)).dt.normalize()
    return out
