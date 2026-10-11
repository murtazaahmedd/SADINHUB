"""The two detectors and the shared alert gates.

A consumption alert needs ALL of:
  1. trustworthy readings for the day (checked in data.py),
  2. enough history to know the meter's usual error (features.py),
  3. a difference big enough to matter:  |kWh difference| >= max(10 kWh, 20% of D),
  4. the detector's own "unusual" test:
       simple rule      : |z| >= 3, i.e. at least 3 times this meter's usual error
       Isolation Forest : anomaly score at or below the cut-off chosen on July-August

Each detector's alerts are kept separately, so the page can show where both
agree, what only the rule catches and what only the ML model catches.

Time order (no future information):
  development  1 Jan - 30 Jun 2026  -> the Isolation Forest is trained here
  validation   1 Jul - 31 Aug 2026  -> its cut-off is set here, to raise as many
                                       alerts as the simple rule does
  final test   1 - 30 Sep 2026      -> untouched until the design was fixed
  latest       1 Oct onwards        -> scored the same way, not part of the test
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .features import feature_matrix

RULE_Z = 3.0
MIN_DIFF_KWH = 10.0
MIN_DIFF_SHARE = 0.20

PERIODS = [
    ("development", "2026-01-01", "2026-06-30"),
    ("validation", "2026-07-01", "2026-08-31"),
    ("final test", "2026-09-01", "2026-09-30"),
    ("latest", "2026-10-01", "2026-12-31"),
]
FOREST = dict(n_estimators=400, max_samples=256, contamination="auto", random_state=7)


def period_of(dates):
    out = pd.Series("", index=dates.index, dtype=object)
    for name, start, end in PERIODS:
        out[(dates >= start) & (dates <= end)] = name
    return out


def material(frame):
    return frame["diff_kwh"].abs() >= np.maximum(MIN_DIFF_KWH, MIN_DIFF_SHARE * frame["denominator_kwh"])


def detect(frame):
    """Add rule/ML flags, scores and the chosen cut-off to the scored rows."""
    out = frame.copy()
    out["period"] = period_of(out["date"])
    scored = out["eligible"] & out["history_ok"]
    out["material"] = scored & material(out)

    dev = scored & (out["period"] == "development")
    forest = IsolationForest(**FOREST).fit(feature_matrix(out[dev]))
    out["ml_score"] = np.nan
    out.loc[scored, "ml_score"] = forest.score_samples(feature_matrix(out[scored]))

    out["rule_flag"] = out["material"] & (out["z"].abs() >= RULE_Z)
    val = out["material"] & (out["period"] == "validation")
    target = int(out.loc[val, "rule_flag"].sum())
    val_scores = np.sort(out.loc[val, "ml_score"].to_numpy())
    cutoff = float(val_scores[target - 1]) if target > 0 else float(val_scores.min()) - 1.0
    out["ml_flag"] = out["material"] & (out["ml_score"] <= cutoff)

    out["direction"] = np.where(out["diff_kwh"] > 0, "higher", "lower")
    out["detector"] = np.select(
        [out["rule_flag"] & out["ml_flag"], out["rule_flag"], out["ml_flag"]],
        ["both", "rule", "ml"], default="")
    settings = {"rule_z": RULE_Z, "min_diff_kwh": MIN_DIFF_KWH, "min_diff_share": MIN_DIFF_SHARE,
                "ml_cutoff": cutoff, "validation_rule_alerts": target,
                "forest": {k: v for k, v in FOREST.items()}, "trained_rows": int(dev.sum())}
    return out, forest, settings


def rescore(rows, forest, cutoff):
    """Apply both detectors to rows whose features were recomputed (synthetic check)."""
    mat = material(rows)
    score = forest.score_samples(feature_matrix(rows))
    return mat & (rows["z"].abs() >= RULE_Z), mat & (score <= cutoff)
