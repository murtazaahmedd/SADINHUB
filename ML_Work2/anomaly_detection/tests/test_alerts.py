"""Checks for the consumption alerts.

Run from ML_Work2:  .venv\\Scripts\\python.exe -m pytest anomaly_detection/tests -q
"""
import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore", category=RuntimeWarning)

from anomaly_detection.data import load_inputs, meter_days  # noqa: E402
from anomaly_detection.detectors import detect  # noqa: E402
from anomaly_detection.features import FEATURES, build_features  # noqa: E402
from anomaly_detection.quality import historical_register_check, quality_alerts  # noqa: E402
from anomaly_detection.report import alert_table  # noqa: E402


@pytest.fixture(scope="module")
def scored():
    daily, _ = load_inputs()
    frame = meter_days()
    out, forest, settings = detect(build_features(frame, daily))
    return daily, frame, out, settings


def test_one_same_day_forecast_per_meter_day(scored):
    _, frame, _, _ = scored
    assert not frame.duplicated(["meter_label", "date"]).any()
    assert frame["expected_kwh"].notna().all()
    assert frame.attrs["mapping_rows_checked"] > 7000


def test_features_never_use_later_days(scored):
    """Scramble every reading after a cut-off day: features up to that day must not move."""
    daily, frame, _, _ = scored
    cut = pd.Timestamp("2026-06-15")
    rng = np.random.default_rng(1)
    later = daily["date"] > cut
    scrambled_daily = daily.copy()
    scrambled_daily.loc[later, "gold_kwh"] = rng.uniform(0, 900, later.sum())
    scrambled_frame = frame.copy()
    later_f = scrambled_frame["date"] > cut
    scrambled_frame.loc[later_f, "actual_kwh"] = rng.uniform(0, 900, later_f.sum())
    a = build_features(frame, daily)
    b = build_features(scrambled_frame, scrambled_daily)
    upto = a["date"] <= cut
    pd.testing.assert_frame_equal(a.loc[upto, FEATURES + ["usual_error_kwh"]], b.loc[upto, FEATURES + ["usual_error_kwh"]])


def test_untrusted_days_are_never_consumption_alerts(scored):
    _, _, out, _ = scored
    flagged = out[out["detector"] != ""]
    assert flagged["eligible"].all()
    assert out.loc[~out["eligible"], "actual_kwh"].isna().all()


def test_ml_cutoff_matches_rule_volume_on_validation_only(scored):
    _, _, out, settings = scored
    val = out[out["period"] == "validation"]
    assert int(val["rule_flag"].sum()) == int(val["ml_flag"].sum()) == settings["validation_rule_alerts"]


def test_no_infinite_percentages_and_switch_on_still_caught(scored):
    _, _, out, _ = scored
    alerts = alert_table(out)
    assert np.isfinite(alerts["pct_of_expected"].dropna()).all()
    zn16 = alerts[(alerts["meter_label"] == "ZN.16-PB_child__1_ch3pm") & (alerts["date"].dt.strftime("%Y-%m") == "2026-06")]
    assert len(zn16) > 0 and (zn16["direction"] == "higher").all()


def test_one_outage_is_never_split_by_a_site_wide_day(scored):
    """A meter already offline when a site-wide day starts keeps one continuous episode."""
    _, _, out, _ = scored
    q = quality_alerts(out)
    zn9 = q[(q["meter_label"] == "ZN.9-PB_child__1_ch3pm") & (q["start"] <= "2026-06-27") & (q["end"] >= "2026-06-16")]
    assert len(zn9) == 1
    assert zn9["catchup_kwh"].iloc[0] > 0                      # the catch-up stays with the outage it belongs to
    site = q[(q["scope"] == "site") & (q["start"] == "2026-06-26")]
    assert len(site) == 1 and site["meters"].iloc[0] < out["meter_label"].nunique()


def test_less_frequent_readings_are_not_missing_hours(scored):
    _, _, out, _ = scored
    q = quality_alerts(out)
    slow = q[q["title"] == "Readings less frequent than normal"]
    assert len(slow) > 0 and (slow["missing_hours"] == 0).all()


def test_known_events_use_direction_and_onset(scored):
    from anomaly_detection.report import evaluate
    daily, _, out, settings = scored
    from sklearn.ensemble import IsolationForest  # noqa: F401  (forest rebuilt inside detect)
    _, forest, settings = detect(out[[c for c in out.columns if c not in
                                      ("period", "material", "ml_score", "rule_flag", "ml_flag", "direction", "detector")]])
    ev = evaluate(out, quality_alerts(out), forest, settings, historical_register_check(daily))
    by_meter = {e["meter"]: e for e in ev["known_events"]}
    zn8 = by_meter["ZN.8-PB.2_child__1_ch3pm"]
    assert zn8["first_alert"] == "2026-06-27" and zn8["rule"] >= 1
    for label, s in ev["synthetic_check"].items():
        assert 0 <= s["either"] <= 100


def test_quality_episodes_and_historical_register_check(scored):
    daily, _, out, _ = scored
    q = quality_alerts(out)
    assert {"meter", "site"} <= set(q["scope"])
    assert (q["end"] >= q["start"]).all()
    handover = q[q["title"] == "Data handover gap"]
    assert len(handover) == 1
    reg = historical_register_check(daily)
    assert reg["flagged_as_data_problem"] and reg["reason"] == "register problem"
