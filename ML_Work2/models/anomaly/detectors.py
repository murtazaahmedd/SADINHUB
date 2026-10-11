"""The two detectors and the shared alert gates.

A consumption alert needs ALL of:
  1. trustworthy readings for the day (checked when the meter-day frame is built),
  2. enough history to know the meter's usual error (features),
  3. a difference big enough to matter:  |kWh difference| >= max(10 kWh, 20% of D),
  4. the detector's own "unusual" test:
       simple rule      : |z| >= 3, i.e. at least 3 times this meter's usual error
       Isolation Forest : anomaly score at or below the cut-off chosen on the validation period

Each detector's alerts are kept separately, so the page can show where both
agree, what only the rule catches and what only the ML model catches.

Time order (no future information): the Isolation Forest is trained on the
development period (every replayed day before July), its cut-off is set on the
validation period so it flags a fixed share of checked days (a staffing choice,
not tied to the rule), and the final test period is untouched until the design is fixed.
"""
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from constants.anomaly import (DEVELOPMENT_PERIOD, FOREST, MIN_DIFF_KWH, MIN_DIFF_SHARE, ML_TARGET_ALERT_RATE,
                               PERIOD_PLAN, RULE_Z, VALIDATION_PERIOD)
from errors import Errors
from models.anomaly.features import feature_matrix
from utilities.guard import require, traced

Period = Tuple[str, str, str]


@traced
def period_plan(year: int, first_day: Optional[date] = None) -> List[Period]:
    """
    #Feature: Anomaly detection
    #Scenario: The development / validation / final test / latest periods of a year
    #Complexity: Time O(p) - periods

    Args:
        year: The forecast year.
        first_day: First replayed day. Development starts there, so earlier history trains the detector.

    Returns:
        (name, first day, last day) per period as ISO dates.

    Raises:
        ValueError: When year is None.
    """
    require("period_plan", "year", year)
    plan = [(name, date(year, m1, d1).isoformat(), date(year, m2, d2).isoformat())
            for name, m1, d1, m2, d2 in PERIOD_PLAN]
    if first_day is not None and first_day.isoformat() < plan[0][1]:
        plan[0] = (plan[0][0], first_day.isoformat(), plan[0][2])
    return plan


@traced
def period_of(dates: pd.Series, plan: List[Period]) -> pd.Series:
    """
    #Feature: Anomaly detection
    #Scenario: Name the period each date falls in
    #Complexity: Time O(n*p) - dates times periods

    Args:
        dates: Datetime series.
        plan: Output of period_plan.

    Returns:
        Series of period names ("" outside every period).

    Raises:
        ValueError: When an argument is None.
    """
    require("period_of", "dates", dates)
    require("period_of", "plan", plan)
    out = pd.Series("", index=dates.index, dtype=object)
    for name, start, end in plan:
        out[(dates >= start) & (dates <= end)] = name
    return out


@traced
def material(frame: pd.DataFrame) -> pd.Series:
    """
    #Feature: Anomaly detection
    #Scenario: Is the kWh difference big enough to matter?
    #Complexity: Time O(n) - rows

    Args:
        frame: Rows with diff_kwh and denominator_kwh.

    Returns:
        Boolean series.

    Raises:
        ValueError: When frame is None.
    """
    require("material", "frame", frame)
    return frame["diff_kwh"].abs() >= np.maximum(MIN_DIFF_KWH, MIN_DIFF_SHARE * frame["denominator_kwh"])


@traced
def _ml_cutoff(out: pd.DataFrame, target: int) -> float:
    """
    #Feature: Anomaly detection
    #Scenario: The Isolation Forest score cut-off that flags a fixed share of the validation period
    #Complexity: Time O(n log n) - sort

    Args:
        out: Scored rows with period, material and ml_score.
        target: Number of alerts to allow in the validation period.

    Returns:
        The cut-off; -inf (ML alerts off) when the validation period has no scored rows.

    Raises:
        ValueError: When out is None.
    """
    require("_ml_cutoff", "out", out)
    val = out["material"] & (out["period"] == VALIDATION_PERIOD)
    scores = np.sort(out.loc[val, "ml_score"].to_numpy())
    if scores.size == 0:
        return float("-inf")
    return float(scores[min(target, scores.size) - 1]) if target > 0 else float(scores.min()) - 1.0


@traced
def detect(frame: pd.DataFrame, plan: List[Period]) -> Tuple[pd.DataFrame, IsolationForest, Dict]:
    """
    #Feature: Anomaly detection
    #Scenario: Add rule/ML flags, scores and the chosen cut-off to the scored rows
    #Complexity: Time O(n * trees) - rows times forest size

    Args:
        frame: Output of build_features.
        plan: Output of period_plan.

    Returns:
        (scored rows, fitted forest, settings dict).

    Raises:
        ValueError: When an argument is None.
        Errors.Misc.InsufficientData: When the development period has no scored rows to train on.
    """
    require("detect", "frame", frame)
    require("detect", "plan", plan)
    out = frame.copy()
    out["period"] = period_of(out["date"], plan)
    scored = out["eligible"] & out["history_ok"]
    out["material"] = scored & material(out)
    dev = scored & (out["period"] == DEVELOPMENT_PERIOD)
    if not dev.any():
        raise Errors.Misc.InsufficientData(
            "[detect] The development period has no scored meter-days to train the Isolation Forest. Invariant "
            "violated: at least one eligible day with enough history in the development period. Received: 0 rows. "
            "Fix: run the forecasting pipeline over the full development period first.")
    forest = IsolationForest(**FOREST).fit(feature_matrix(out[dev]))
    out["ml_score"] = np.nan
    out.loc[scored, "ml_score"] = forest.score_samples(feature_matrix(out[scored]))
    out["rule_flag"] = out["material"] & (out["z"].abs() >= RULE_Z)
    checked_validation = int((scored & (out["period"] == VALIDATION_PERIOD)).sum())
    target = int(round(ML_TARGET_ALERT_RATE * checked_validation))
    cutoff = _ml_cutoff(out, target)
    out["ml_flag"] = out["material"] & (out["ml_score"] <= cutoff)
    out["direction"] = np.where(out["diff_kwh"] > 0, "higher", "lower")
    out["detector"] = np.select([out["rule_flag"] & out["ml_flag"], out["rule_flag"], out["ml_flag"]],
                                ["both", "rule", "ml"], default="")
    settings = {"rule_z": RULE_Z, "min_diff_kwh": MIN_DIFF_KWH, "min_diff_share": MIN_DIFF_SHARE,
                "ml_cutoff": cutoff if np.isfinite(cutoff) else None, "ml_target_alert_rate": ML_TARGET_ALERT_RATE,
                "validation_checked_days": checked_validation, "validation_ml_target": target,
                "validation_rule_alerts": int((out["rule_flag"] & (out["period"] == VALIDATION_PERIOD)).sum()),
                "forest": dict(FOREST), "trained_rows": int(dev.sum())}
    return out, forest, settings


@traced
def rescore(rows: pd.DataFrame, forest: IsolationForest, cutoff) -> Tuple[pd.Series, pd.Series]:
    """
    #Feature: Anomaly detection
    #Scenario: Apply both detectors to rows whose features were recomputed (synthetic check)
    #Complexity: Time O(n * trees) - rows times forest size

    Args:
        rows: Rows with recomputed features.
        forest: The fitted forest.
        cutoff: Score cut-off (None means ML alerts are off).

    Returns:
        (rule flags, ML flags).

    Raises:
        ValueError: When an argument is None.
    """
    require("rescore", "rows", rows)
    require("rescore", "forest", forest)
    mat = material(rows)
    score = forest.score_samples(feature_matrix(rows))
    limit = float("-inf") if cutoff is None else cutoff
    return mat & (rows["z"].abs() >= RULE_Z), mat & (score <= limit)
