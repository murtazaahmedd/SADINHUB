"""Compare forecasts with gold-booked actuals and compute accuracy.

Accuracy = 100 x (1 - sum|forecast - actual| / sum actual), i.e. 100 - WAPE,
the same measure for a day, a 7-day total and a 30-day total.

What is compared, and why some periods are left out:
  * 1 day   - only days whose readings cover >= 95% of the day with no outage
              catch-up booked into them (otherwise gold's daily number is not
              the energy used that day; e.g. 201 days of energy on 2025-12-05).
  * 7 / 30  - the period total from gold. Gaps *inside* the period do not
              matter (the register keeps counting); it is only left out when
              an outage of over an hour spans its first or last midnight,
              which would move energy across the period boundary.
Zones and the facility are scored on the meters measurable in that period.
"""
import numpy as np
import pandas as pd

from . import config


def _gold_cumsum(panel):
    ok = ~np.isnan(panel.gold)
    cs = np.concatenate([np.zeros((panel.gold.shape[0], 1)), np.cumsum(np.where(ok, panel.gold, 0), 1)], 1)
    cn = np.concatenate([np.zeros((panel.gold.shape[0], 1)), np.cumsum(ok, 1)], 1)
    return cs, cn


def meter_periods(panel, preds, horizons=config.HORIZONS):
    """One row per meter x issue day x horizon with forecast, actual and flags."""
    cs, cn = _gold_cumsum(panel)
    n = panel.gold.shape[1]
    frames = []
    for h in horizons:
        part = preds[preds["days_ahead"] < h]
        agg = part.groupby(["meter_idx", "origin_idx"]).agg(forecast=("pred_kwh", "sum"),
                                                             days=("pred_kwh", "size")).reset_index()
        agg = agg[agg["days"] == h]
        m = agg["meter_idx"].to_numpy()
        t = agg["origin_idx"].to_numpy()
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
        frames.append(agg.drop(columns="days"))
    out = pd.concat(frames, ignore_index=True)
    out["meter_label"] = np.array(panel.meters)[out["meter_idx"]]
    out["zone"] = panel.zone[out["meter_idx"]]
    out["issue_date"] = panel.dates[out["origin_idx"].to_numpy()]
    return out


def calendar_months(panel, preds):
    """Whole-month forecast issued on the 1st (data to the last day of the
    previous month) against the month's gold total - one row per meter-month."""
    cs, cn = _gold_cumsum(panel)
    n = panel.gold.shape[1]
    firsts = [d for d in panel.dates if d.day == 1]
    frames = []
    for first in firsts:
        t = panel.index_of(first)
        h = first.days_in_month
        part = preds[(preds["origin_idx"] == t) & (preds["days_ahead"] < h)]
        if part.empty:
            continue
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
        frames.append(agg.drop(columns="days"))
    out = pd.concat(frames, ignore_index=True)
    out["meter_label"] = np.array(panel.meters)[out["meter_idx"]]
    out["zone"] = panel.zone[out["meter_idx"]]
    out["issue_date"] = panel.dates[out["origin_idx"].to_numpy()]
    return out


def all_levels(meter_rows):
    """Stack meter, zone and facility rows (measurable periods only)."""
    m = meter_rows[meter_rows["measurable"]].copy()
    m["level"] = "meter"
    m["unit"] = m["meter_label"]
    zone = (m.groupby(["zone", "issue_date", "horizon"])
            .agg(forecast=("forecast", "sum"), actual=("actual", "sum"), meters=("meter_idx", "size"))
            .reset_index())
    zone["level"] = "zone"
    zone["unit"] = "Zone " + zone["zone"].astype(str).str.zfill(2)
    fac = (m.groupby(["issue_date", "horizon"])
           .agg(forecast=("forecast", "sum"), actual=("actual", "sum"), meters=("meter_idx", "size"))
           .reset_index())
    fac["level"] = "facility"
    fac["unit"] = "Facility (all zones)"
    m["meters"] = 1
    cols = ["level", "unit", "issue_date", "horizon", "forecast", "actual", "meters"]
    out = pd.concat([m[cols], zone[cols], fac[cols]], ignore_index=True)
    out["abs_error"] = (out["forecast"] - out["actual"]).abs()
    out["accuracy"] = 100.0 * (1.0 - out["abs_error"] / out["actual"])
    return out


def accuracy(frame):
    total = frame["actual"].sum()
    return float(100.0 * (1.0 - frame["abs_error"].sum() / total)) if total > 0 else np.nan


def month_matrix(levels, level, horizon):
    """Unit x month accuracy (pooled over every issue day in the month)."""
    part = levels[(levels["level"] == level) & (levels["horizon"] == horizon)].copy()
    part["month"] = part["issue_date"].dt.strftime("%Y-%m")
    grouped = part.groupby(["unit", "month"])
    table = grouped.apply(accuracy, include_groups=False).unstack("month")
    return table.sort_index(key=lambda s: s.map(_unit_sort_key))


def _unit_sort_key(unit):
    import re
    nums = [int(x) for x in re.findall(r"\d+", unit)[:2]]
    return tuple(nums + [0] * (2 - len(nums))) if nums else (999, 0)


def summary(levels):
    rows = []
    for (level, horizon), part in levels.groupby(["level", "horizon"], sort=False):
        part = part.copy()
        part["month"] = part["issue_date"].dt.strftime("%Y-%m")
        cells = part.groupby(["unit", "month"]).apply(accuracy, include_groups=False)
        units = part.groupby("unit").apply(accuracy, include_groups=False)
        rows.append({
            "level": level, "horizon": horizon,
            "overall_accuracy": round(accuracy(part), 2),
            "units": int(units.size),
            "units_at_or_above_80": int((units >= config.ACCURACY_TARGET).sum()),
            "lowest_unit_accuracy": round(float(units.min()), 2),
            "unit_month_cells": int(cells.size),
            "cells_at_or_above_80": int((cells >= config.ACCURACY_TARGET).sum()),
            "lowest_cell": round(float(cells.min()), 2),
            "periods_compared": int(len(part)),
        })
    return pd.DataFrame(rows)
