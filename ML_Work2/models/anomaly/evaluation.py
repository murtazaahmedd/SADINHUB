"""Evaluation of the detectors: alerts per week, agreement, known events and a synthetic fault check.

There is no record of confirmed faults, so no "accuracy %" is claimed. The synthetic check changes each checked
final-test meter-day on its own to 60% above (or below) its real reading, with its history and that morning's
forecast left as they were. Only flags in the same direction as the change count. It only proves the system
reacts to a sudden change; it says nothing about how well real equipment problems are found.
"""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from constants.anomaly import FINAL_TEST_PERIOD, SYNTHETIC_FACTORS
from models.anomaly.detectors import Period, rescore
from models.anomaly.features import with_actual
from utilities.anomaly_report import event_rows, period_rows, site_event_rows
from utilities.guard import require, traced


@traced
def synthetic_check(scored: pd.DataFrame, forest, cutoff: Optional[float]) -> Dict:
    """
    #Feature: Alert evaluation
    #Scenario: Does the system react to a sudden 60% jump or dip on a real day?
    #Complexity: Time O(n * trees) - rows times forest size

    Args:
        scored: Scored rows.
        forest: The fitted Isolation Forest.
        cutoff: ML score cut-off (None means ML alerts are off).

    Returns:
        label -> detection rates; empty when the final test period has no checked days.

    Raises:
        ValueError: When scored is None.
    """
    require("synthetic_check", "scored", scored)
    rows = scored[(scored["period"] == FINAL_TEST_PERIOD) & scored["eligible"] & scored["history_ok"]]
    if rows.empty:
        return {}
    synthetic = {}
    for label, factor in SYNTHETIC_FACTORS.items():
        moved = with_actual(rows, rows["actual_kwh"] * factor)
        rule, ml = rescore(moved, forest, cutoff)
        right_way = np.sign(moved["diff_kwh"]) == np.sign(factor - 1)   # flag must match the change
        synthetic[label] = {"days_tested": int(len(moved)),
                            "rule": round(float((rule & right_way).mean()) * 100, 1),
                            "ml": round(float((ml & right_way).mean()) * 100, 1),
                            "either": round(float(((rule | ml) & right_way).mean()) * 100, 1),
                            "wrong_direction_flags": int(((rule | ml) & ~right_way).sum())}
    return synthetic


@traced
def evaluate(scored: pd.DataFrame, quality: pd.DataFrame, forest, settings: Dict, register_check: Dict,
             plan: List[Period]) -> Dict:
    """
    #Feature: Alert evaluation
    #Scenario: Everything measured about the detectors
    #Complexity: Time O(n * trees) - rows times forest size

    Args:
        scored: Scored rows.
        quality: Data-problem episodes.
        forest: The fitted Isolation Forest.
        settings: Detector settings (includes ml_cutoff).
        register_check: Output of historical_register_check.
        plan: Period plan.

    Returns:
        Dict with periods, final_test_agreement, known_events, synthetic_check, register check and settings.

    Raises:
        ValueError: When an argument is None.
    """
    require("evaluate", "scored", scored)
    require("evaluate", "settings", settings)
    alerts = scored[scored["detector"] != ""]
    test = alerts[alerts["period"] == FINAL_TEST_PERIOD]
    agreement = {"both": int((test["detector"] == "both").sum()), "rule_only": int((test["detector"] == "rule").sum()),
                 "ml_only": int((test["detector"] == "ml").sum())}
    return {"periods": period_rows(scored, alerts, quality, plan), "final_test_agreement": agreement,
            "known_events": event_rows(alerts) + site_event_rows(quality, plan),
            "synthetic_check": synthetic_check(scored, forest, settings["ml_cutoff"]),
            "historical_register_check": register_check, "settings": settings}
