"""Features for each checkable meter-day, using only days before it.

For a meter-day with actual A and expected E (the forecast made that morning):

  difference        e   = A - E                                    (kWh)
  size of normal    M28 = median actual kWh over the previous 28 days (trusted days only)
  denominator       D   = max(E, 0.25 x M28, 5 kWh)    -> no huge % when E is near zero
  relative          rel = e / D
  usual error       U   = 1.4826 x median |A - E| over this meter's last 28 checked
                          days (at least 14 needed), never below max(2 kWh, 5% of M28)
  standardised      z   = e / U        "how many usual errors away"
  jump              (A - previous trusted day) / max(previous, 0.25 x M28, 5)
  similar days      (A - median of the same weekday in the last 4 weeks) / max(that, 0.25 x M28, 5)
  recent level      (A - M28) / max(M28, 5)

A day's own reading never enters its own history, so nothing from the day being
checked (or later) is used to judge it.
"""
from typing import Dict, List

import numpy as np
import pandas as pd

from constants.anomaly import (CLIP, FEATURES, FLOOR_KWH, LEVEL_DAYS, MAD_SCALE, MIN_LEVEL_DAYS,
                               MIN_RESIDUALS, MIN_USUAL_ERROR_KWH, MIN_USUAL_ERROR_SHARE, RESIDUAL_WINDOW)
from utilities.guard import require, traced


@traced
def trusted_history(daily: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    #Feature: Anomaly features
    #Scenario: Gold kWh on trusted days for every meter, indexed by date
    #Complexity: Time O(n) - rows

    Args:
        daily: The daily table.

    Returns:
        meter label -> Series of gold kWh sorted by date.

    Raises:
        ValueError: When daily is None.
    """
    require("trusted_history", "daily", daily)
    trusted = daily[daily["measurable_day"]]
    return {m: g.set_index("date")["gold_kwh"].sort_index() for m, g in trusted.groupby("meter_label")}


@traced
def _recent_level(history: pd.Series, date: pd.Timestamp):
    """
    #Feature: Anomaly features
    #Scenario: The meter's recent trusted days and their median level
    #Complexity: Time O(h) - history length

    Args:
        history: The meter's trusted gold kWh by date.
        date: The day being checked.

    Returns:
        (recent trusted days, median level or NaN).

    Raises:
        ValueError: When an argument is None.
    """
    require("_recent_level", "history", history)
    require("_recent_level", "date", date)
    recent = history[(history.index < date) & (history.index >= date - pd.Timedelta(days=LEVEL_DAYS))]
    if len(recent) < MIN_LEVEL_DAYS:
        recent = history[history.index < date].tail(MIN_LEVEL_DAYS)
    return recent, (float(np.median(recent)) if len(recent) else np.nan)


@traced
def _usual_error(residuals: List[float], level: float):
    """
    #Feature: Anomaly features
    #Scenario: This meter's usual forecast error over its recent checked days
    #Complexity: Time O(w) - residual window

    Args:
        residuals: The meter's earlier (actual - expected) values.
        level: Recent median level (NaN when unknown).

    Returns:
        (usual error in kWh or NaN, whether there was enough history).

    Raises:
        ValueError: When residuals is None.
    """
    require("_usual_error", "residuals", residuals)
    past = residuals[-RESIDUAL_WINDOW:]
    if len(past) >= MIN_RESIDUALS and np.isfinite(level):
        usual = max(MAD_SCALE * float(np.median(np.abs(past))), MIN_USUAL_ERROR_KWH, MIN_USUAL_ERROR_SHARE * level)
        return usual, True
    return np.nan, False


@traced
def _features_for(actual: float, expected: float, date: pd.Timestamp, history: pd.Series,
                  residuals: List[float]) -> Dict:
    """
    #Feature: Anomaly features
    #Scenario: Feature values for one meter-day given that meter's earlier history
    #Complexity: Time O(h) - history length

    Args:
        actual: Gold kWh of the day.
        expected: That morning's forecast.
        date: The day.
        history: The meter's trusted gold kWh by date.
        residuals: The meter's earlier (actual - expected) values.

    Returns:
        Dict of features and the reference values behind them.

    Raises:
        ValueError: When an argument is None.
    """
    require("_features_for", "history", history)
    require("_features_for", "date", date)
    recent, level = _recent_level(history, date)
    floor = max(FLOOR_KWH, 0.25 * level) if np.isfinite(level) else FLOOR_KWH
    denom = max(expected, floor)
    diff = actual - expected
    usual, history_ok = _usual_error(residuals, level)
    before = history[(history.index < date) & (history.index >= date - pd.Timedelta(days=7))]
    prev = float(before.iloc[-1]) if len(before) else np.nan
    prev_date = before.index[-1] if len(before) else pd.NaT   # may be earlier than yesterday
    same_day = recent[recent.index.dayofweek == date.dayofweek]
    similar_ref = float(np.median(same_day)) if len(same_day) >= 2 else level

    def scaled(ref):
        return (actual - ref) / max(ref, floor) if np.isfinite(ref) else 0.0

    return {
        "diff_kwh": diff, "denominator_kwh": denom, "recent_level_kwh": level, "usual_error_kwh": usual,
        "history_ok": history_ok, "floor_kwh": floor, "previous_kwh": prev, "previous_date": prev_date,
        "similar_ref_kwh": similar_ref, "z": diff / usual if history_ok else np.nan, "rel": diff / denom,
        "jump": scaled(prev), "similar": scaled(similar_ref),
        "level": (actual - level) / max(level, FLOOR_KWH) if np.isfinite(level) else 0.0,
        "pct_of_expected": diff / expected if expected >= floor else np.nan,   # shown only when meaningful
    }


@traced
def _meter_feature_rows(part: pd.DataFrame, hist: pd.Series) -> List[Dict]:
    """
    #Feature: Anomaly features
    #Scenario: Feature rows of one meter, in date order, each using only earlier days
    #Complexity: Time O(d*h) - days times history

    Args:
        part: That meter's rows of the meter-day frame.
        hist: That meter's trusted history.

    Returns:
        One dict per row with its index and (when eligible) its features.

    Raises:
        ValueError: When an argument is None.
    """
    require("_meter_feature_rows", "part", part)
    require("_meter_feature_rows", "hist", hist)
    rows, residuals = [], []
    for row in part.itertuples():
        out = {"index": row.Index}
        if row.eligible:
            out.update(_features_for(row.actual_kwh, row.expected_kwh, row.date, hist, residuals))
            residuals.append(row.actual_kwh - row.expected_kwh)   # becomes history only after scoring
        rows.append(out)
    return rows


@traced
def build_features(frame: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """
    #Feature: Anomaly features
    #Scenario: Add feature columns to every eligible row of the meter-day frame
    #Complexity: Time O(m*d*h) - meters, days, history

    Args:
        frame: Output of build_meter_days.
        daily: The daily table.

    Returns:
        The frame joined with its features.

    Raises:
        ValueError: When an argument is None.
    """
    require("build_features", "frame", frame)
    require("build_features", "daily", daily)
    history = trusted_history(daily)
    rows: List[Dict] = []
    for meter, part in frame.groupby("meter_label", sort=False):
        rows += _meter_feature_rows(part, history.get(meter, pd.Series(dtype=float)))
    out = frame.join(pd.DataFrame(rows).set_index("index"))
    out["history_ok"] = out["history_ok"].astype("boolean").fillna(False).astype(bool)
    return out


@traced
def with_actual(rows: pd.DataFrame, new_actual) -> pd.DataFrame:
    """
    #Feature: Anomaly features
    #Scenario: Recompute every feature for a different actual kWh, keeping the same history
    #Complexity: Time O(n) - rows

    Used for the synthetic check: only the day's own reading changes; all
    history-based references (usual error, previous day, recent level) stay.

    Args:
        rows: Scored rows.
        new_actual: Replacement actual kWh per row.

    Returns:
        A copy with recomputed diff, rel, z, jump, similar and level.

    Raises:
        ValueError: When an argument is None.
    """
    require("with_actual", "rows", rows)
    require("with_actual", "new_actual", new_actual)
    out = rows.copy()
    a = np.asarray(new_actual, dtype=float)
    out["actual_kwh"] = a
    out["diff_kwh"] = a - out["expected_kwh"]
    out["rel"] = out["diff_kwh"] / out["denominator_kwh"]
    out["z"] = out["diff_kwh"] / out["usual_error_kwh"]

    def scaled(ref):
        ref = out[ref]
        return np.where(np.isfinite(ref), (a - ref) / np.maximum(ref, out["floor_kwh"]), 0.0)

    out["jump"] = scaled("previous_kwh")
    out["similar"] = scaled("similar_ref_kwh")
    out["level"] = (a - out["recent_level_kwh"]) / np.maximum(out["recent_level_kwh"], FLOOR_KWH)
    return out


@traced
def feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    """
    #Feature: Anomaly features
    #Scenario: Clipped feature matrix for the Isolation Forest
    #Complexity: Time O(n) - rows

    Args:
        frame: Rows with the feature columns.

    Returns:
        Array of the clipped features.

    Raises:
        ValueError: When frame is None.
    """
    require("feature_matrix", "frame", frame)
    x = frame[FEATURES].astype(float).copy()
    for col, limit in CLIP.items():
        x[col] = x[col].clip(-limit, limit)
    return x.to_numpy()
