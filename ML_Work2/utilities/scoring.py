"""Compare forecasts with gold-booked actuals and compute accuracy.

Accuracy = 100 x (1 - sum|forecast - actual| / sum actual), i.e. 100 - WAPE,
the same measure for a day, a 7-day total and a 30-day total.

What is compared, and why some periods are left out:
  * 1 day   - only days whose readings cover >= 95% of the day with no outage
              catch-up booked into them (otherwise gold's daily number is not
              the energy used that day).
  * 7 / 30  - the period total from gold. Gaps *inside* the period do not
              matter (the register keeps counting); it is only left out when an
              outage of over an hour spans its first or last midnight, which
              would move energy across the period boundary.
Zones and the facility are scored on the meters measurable in that period.
"""
import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from constants.forecasting import ACCURACY_TARGET, HORIZONS
from utilities.guard import require, traced


@traced
def _gold_cumsum(panel) -> Tuple[np.ndarray, np.ndarray]:
    """
    #Feature: Accuracy scoring
    #Scenario: Cumulative gold kWh and day counts so any period total is a difference
    #Complexity: Time O(m*n) - meters times days

    Args:
        panel: Panel with the gold array.

    Returns:
        (cumulative kWh, cumulative day count), each with a leading zero column.

    Raises:
        ValueError: When panel is None.
    """
    require("_gold_cumsum", "panel", panel)
    ok = ~np.isnan(panel.gold)
    cs = np.concatenate([np.zeros((panel.gold.shape[0], 1)), np.cumsum(np.where(ok, panel.gold, 0), 1)], 1)
    cn = np.concatenate([np.zeros((panel.gold.shape[0], 1)), np.cumsum(ok, 1)], 1)
    return cs, cn


@traced
def _label_periods(panel, out: pd.DataFrame) -> pd.DataFrame:
    """
    #Feature: Accuracy scoring
    #Scenario: Add meter label, zone and issue date to scored periods
    #Complexity: Time O(n) - linear

    Args:
        panel: Panel with meters, zone and dates.
        out: Scored periods with meter_idx and origin_idx.

    Returns:
        The frame with meter_label, zone and issue_date.

    Raises:
        ValueError: When an argument is None.
    """
    require("_label_periods", "panel", panel)
    require("_label_periods", "out", out)
    out["meter_label"] = np.array(panel.meters)[out["meter_idx"]]
    out["zone"] = panel.zone[out["meter_idx"]]
    out["issue_date"] = panel.dates[out["origin_idx"].to_numpy()]
    return out


@traced
def _horizon_block(panel, preds: pd.DataFrame, h: int, cs: np.ndarray, cn: np.ndarray) -> pd.DataFrame:
    """
    #Feature: Accuracy scoring
    #Scenario: Forecast, actual and measurable flag for every meter and issue day at one horizon
    #Complexity: Time O(rows) - linear

    Args:
        panel: Panel with measurable, outage_start and exists arrays.
        preds: Every-morning forecasts (meter_idx, origin_idx, days_ahead, pred_kwh).
        h: Horizon in days.
        cs: Cumulative gold kWh.
        cn: Cumulative gold day counts.

    Returns:
        One row per meter and issue day with forecast, actual, measurable and horizon.

    Raises:
        ValueError: When an argument is None.
    """
    require("_horizon_block", "panel", panel)
    require("_horizon_block", "preds", preds)
    n = panel.gold.shape[1]
    part = preds[preds["days_ahead"] < h]
    agg = part.groupby(["meter_idx", "origin_idx"]).agg(forecast=("pred_kwh", "sum"),
                                                         days=("pred_kwh", "size")).reset_index()
    agg = agg[agg["days"] == h]
    m, t = agg["meter_idx"].to_numpy(), agg["origin_idx"].to_numpy()
    end = np.minimum(t + h, n)
    actual = cs[m, end] - cs[m, t]
    complete = (cn[m, end] - cn[m, t] == h) & (t + h <= n)
    if h == 1:
        ok = complete & panel.measurable[m, t]
    else:
        boundary_ok = ~panel.outage_start[m, t] & ~panel.outage_start[m, np.minimum(t + h, n - 1)]
        ok = complete & boundary_ok & panel.exists[m, t] & (actual > 0)
    agg["horizon"] = h
    agg["actual"] = np.where(complete, actual, np.nan)
    agg["measurable"] = ok
    return agg.drop(columns="days")


@traced
def meter_periods(panel, preds: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    """
    #Feature: Accuracy scoring
    #Scenario: One row per meter x issue day x horizon with forecast, actual and flags
    #Complexity: Time O(rows * horizons) - linear

    Args:
        panel: The panel.
        preds: Every-morning forecasts.
        horizons: Horizons in days.

    Returns:
        Scored periods with meter_label, zone and issue_date.

    Raises:
        ValueError: When panel or preds is None.
    """
    require("meter_periods", "panel", panel)
    require("meter_periods", "preds", preds)
    cs, cn = _gold_cumsum(panel)
    out = pd.concat([_horizon_block(panel, preds, h, cs, cn) for h in horizons], ignore_index=True)
    return _label_periods(panel, out)


@traced
def _month_block(panel, preds: pd.DataFrame, first: pd.Timestamp, cs: np.ndarray, cn: np.ndarray) -> Optional[pd.DataFrame]:
    """
    #Feature: Accuracy scoring
    #Scenario: A whole-month forecast issued on the 1st against that month's gold total
    #Complexity: Time O(rows) - linear

    Args:
        panel: The panel.
        preds: Every-morning forecasts.
        first: First day of the month.
        cs: Cumulative gold kWh.
        cn: Cumulative gold day counts.

    Returns:
        One row per meter, or None when no forecast was issued that day.

    Raises:
        ValueError: When an argument is None.
    """
    require("_month_block", "panel", panel)
    require("_month_block", "first", first)
    n = panel.gold.shape[1]
    t, h = panel.index_of(first), first.days_in_month
    part = preds[(preds["origin_idx"] == t) & (preds["days_ahead"] < h)]
    if part.empty:
        return None
    agg = part.groupby("meter_idx").agg(forecast=("pred_kwh", "sum"), days=("pred_kwh", "size")).reset_index()
    agg = agg[agg["days"] == h]
    m = agg["meter_idx"].to_numpy()
    end = min(t + h, n)
    complete = (cn[m, end] - cn[m, t] == h) & (t + h <= n)
    actual = cs[m, end] - cs[m, t]
    boundary_ok = ~panel.outage_start[m, t] & ~panel.outage_start[m, min(t + h, n - 1)]
    agg["actual"] = np.where(complete, actual, np.nan)
    agg["measurable"] = complete & boundary_ok & panel.exists[m, t] & (actual > 0)
    agg["horizon"] = "month"
    agg["origin_idx"] = t
    return agg.drop(columns="days")


@traced
def calendar_months(panel, preds: pd.DataFrame) -> pd.DataFrame:
    """
    #Feature: Accuracy scoring
    #Scenario: Whole-month forecasts issued on the 1st, one row per meter-month
    #Complexity: Time O(months * rows) - linear

    Args:
        panel: The panel.
        preds: Every-morning forecasts.

    Returns:
        Scored months with meter_label, zone and issue_date.

    Raises:
        ValueError: When panel or preds is None.
    """
    require("calendar_months", "panel", panel)
    require("calendar_months", "preds", preds)
    cs, cn = _gold_cumsum(panel)
    blocks = [_month_block(panel, preds, d, cs, cn) for d in panel.dates if d.day == 1]
    out = pd.concat([b for b in blocks if b is not None], ignore_index=True)
    return _label_periods(panel, out)


@traced
def _zone_and_facility(m: pd.DataFrame) -> pd.DataFrame:
    """
    #Feature: Accuracy scoring
    #Scenario: Sum measurable meter rows into zone and facility rows
    #Complexity: Time O(n) - linear

    Args:
        m: Measurable meter rows with level, unit and meters columns.

    Returns:
        Zone and facility rows in the same columns.

    Raises:
        ValueError: When m is None.
    """
    require("_zone_and_facility", "m", m)
    agg = dict(forecast=("forecast", "sum"), actual=("actual", "sum"), meters=("meter_idx", "size"))
    zone = m.groupby(["zone", "issue_date", "horizon"]).agg(**agg).reset_index()
    zone["level"] = "zone"
    zone["unit"] = "Zone " + zone["zone"].astype(str).str.zfill(2)
    fac = m.groupby(["issue_date", "horizon"]).agg(**agg).reset_index()
    fac["level"] = "facility"
    fac["unit"] = "Facility (all zones)"
    cols = ["level", "unit", "issue_date", "horizon", "forecast", "actual", "meters"]
    return pd.concat([zone[cols], fac[cols]], ignore_index=True)


@traced
def all_levels(meter_rows: pd.DataFrame) -> pd.DataFrame:
    """
    #Feature: Accuracy scoring
    #Scenario: Stack meter, zone and facility rows (measurable periods only)
    #Complexity: Time O(n) - linear

    Args:
        meter_rows: Output of meter_periods / calendar_months with readable horizon labels.

    Returns:
        Rows with level, unit, forecast, actual, abs_error and accuracy.

    Raises:
        ValueError: When meter_rows is None.
    """
    require("all_levels", "meter_rows", meter_rows)
    m = meter_rows[meter_rows["measurable"]].copy()
    m["level"] = "meter"
    m["unit"] = m["meter_label"]
    upper = _zone_and_facility(m)
    m["meters"] = 1
    cols = ["level", "unit", "issue_date", "horizon", "forecast", "actual", "meters"]
    out = pd.concat([m[cols], upper], ignore_index=True)
    out["abs_error"] = (out["forecast"] - out["actual"]).abs()
    out["accuracy"] = 100.0 * (1.0 - out["abs_error"] / out["actual"])
    return out


@traced
def accuracy(frame: pd.DataFrame) -> float:
    """
    #Feature: Accuracy scoring
    #Scenario: Pooled accuracy = 100 - WAPE
    #Complexity: Time O(n) - linear

    Args:
        frame: Rows with actual and abs_error.

    Returns:
        Accuracy in percent, NaN when total actual is zero.

    Raises:
        ValueError: When frame is None.
    """
    require("accuracy", "frame", frame)
    total = frame["actual"].sum()
    return float(100.0 * (1.0 - frame["abs_error"].sum() / total)) if total > 0 else np.nan


@traced
def _unit_sort_key(unit: str) -> tuple:
    """
    #Feature: Accuracy scoring
    #Scenario: Sort units by zone then meter number
    #Complexity: Time O(len(unit)) - linear

    Args:
        unit: Unit name such as "Zone 03" or "ZN.3-PB.1_child__1_ch3pm".

    Returns:
        A sortable tuple of the first two numbers in the name.

    Raises:
        ValueError: When unit is None.
    """
    require("_unit_sort_key", "unit", unit)
    nums = [int(x) for x in re.findall(r"\d+", unit)[:2]]
    return tuple(nums + [0] * (2 - len(nums))) if nums else (999, 0)


@traced
def month_matrix(levels: pd.DataFrame, level: str, horizon: str) -> pd.DataFrame:
    """
    #Feature: Accuracy scoring
    #Scenario: Unit x month accuracy pooled over every issue day in the month
    #Complexity: Time O(n) - linear

    Args:
        levels: Output of all_levels.
        level: "meter", "zone" or "facility".
        horizon: Horizon label.

    Returns:
        DataFrame indexed by unit with one column per month.

    Raises:
        ValueError: When an argument is None.
    """
    require("month_matrix", "levels", levels)
    part = levels[(levels["level"] == level) & (levels["horizon"] == horizon)].copy()
    part["month"] = part["issue_date"].dt.strftime("%Y-%m")
    table = part.groupby(["unit", "month"]).apply(accuracy, include_groups=False).unstack("month")
    return table.sort_index(key=lambda s: s.map(_unit_sort_key))


@traced
def _summary_row(level: str, horizon: str, part: pd.DataFrame) -> dict:
    """
    #Feature: Accuracy scoring
    #Scenario: One summary line for a level and horizon
    #Complexity: Time O(n) - linear

    Args:
        level: Level name.
        horizon: Horizon label.
        part: Rows of that level and horizon.

    Returns:
        Summary dict with overall accuracy and counts at or above the target.

    Raises:
        ValueError: When part is None.
    """
    require("_summary_row", "part", part)
    part = part.copy()
    part["month"] = part["issue_date"].dt.strftime("%Y-%m")
    cells = part.groupby(["unit", "month"]).apply(accuracy, include_groups=False)
    units = part.groupby("unit").apply(accuracy, include_groups=False)
    return {
        "level": level, "horizon": horizon, "overall_accuracy": round(accuracy(part), 2),
        "units": int(units.size), "units_at_or_above_80": int((units >= ACCURACY_TARGET).sum()),
        "lowest_unit_accuracy": round(float(units.min()), 2), "unit_month_cells": int(cells.size),
        "cells_at_or_above_80": int((cells >= ACCURACY_TARGET).sum()), "lowest_cell": round(float(cells.min()), 2),
        "periods_compared": int(len(part)),
    }


@traced
def summary(levels: pd.DataFrame) -> pd.DataFrame:
    """
    #Feature: Accuracy scoring
    #Scenario: Accuracy summary per level and horizon
    #Complexity: Time O(n) - linear

    Args:
        levels: Output of all_levels.

    Returns:
        One row per level and horizon.

    Raises:
        ValueError: When levels is None.
    """
    require("summary", "levels", levels)
    return pd.DataFrame([_summary_row(level, horizon, part)
                         for (level, horizon), part in levels.groupby(["level", "horizon"], sort=False)])
