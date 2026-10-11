"""Checks for the consumption alerts and data-problem alerts.

They read the saved forecasting outputs, so they are skipped until run_pipeline.py has run.
Checks that depend on a particular period of data skip when that period is not in the data.
"""
import numpy as np
import pandas as pd
import pytest

from constants.anomaly import FEATURES
from constants.known_events import KNOWN_REGISTER_JUMP
from models.anomaly.detectors import detect
from models.anomaly.features import build_features
from tests.conftest import ROOT, alert_inputs
from utilities.anomaly_quality import historical_register_check, quality_alerts
from models.anomaly.evaluation import evaluate
from utilities.anomaly_report import alert_table

OUT = ROOT / "outputs"
needs_outputs = pytest.mark.skipif(
    not all((OUT / f).exists() for f in ("data/daily_actuals.parquet", "validation/every_morning_forecast.parquet",
                                         "data/validated_intervals.parquet")),
    reason="run run_pipeline.py first")


@pytest.fixture(scope="module")
def scored():
    daily, frame, plan = alert_inputs()
    out, forest, settings = detect(build_features(frame, daily), plan)
    return daily, frame, out, forest, settings, plan


@needs_outputs
def test_one_same_day_forecast_per_meter_day(scored):
    _, frame, _, _, _, _ = scored
    assert not frame.duplicated(["meter_label", "date"]).any()
    assert frame["expected_kwh"].notna().all()
    assert frame.attrs["mapping_rows_checked"] > 0.5 * len(frame)


@needs_outputs
def test_features_never_use_later_days(scored):
    """Scramble every reading after a cut-off day: features up to that day must not move."""
    daily, frame, _, _, _, _ = scored
    cut = frame["date"].sort_values().iloc[int(0.8 * len(frame))]
    rng = np.random.default_rng(1)
    later = daily["date"] > cut
    scrambled_daily = daily.copy()
    scrambled_daily.loc[later, "gold_kwh"] = rng.uniform(0, 900, later.sum())
    scrambled_frame = frame.copy()
    later_f = scrambled_frame["date"] > cut
    scrambled_frame.loc[later_f, "actual_kwh"] = rng.uniform(0, 900, later_f.sum())
    a, b = build_features(frame, daily), build_features(scrambled_frame, scrambled_daily)
    upto = a["date"] <= cut
    pd.testing.assert_frame_equal(a.loc[upto, FEATURES + ["usual_error_kwh"]], b.loc[upto, FEATURES + ["usual_error_kwh"]])


@needs_outputs
def test_untrusted_days_are_never_consumption_alerts(scored):
    _, _, out, _, _, _ = scored
    assert out[out["detector"] != ""]["eligible"].all()
    assert out.loc[~out["eligible"], "actual_kwh"].isna().all()


@needs_outputs
def test_ml_alert_level_is_a_fixed_share_of_checked_validation_days(scored):
    """The ML cut-off is a staffing choice (a fixed share), not tied to how many alerts the rule raised."""
    _, _, out, _, settings, _ = scored
    val = out[out["period"] == "validation"]
    if val.empty:
        pytest.skip("no validation period in the data yet")
    checked = int((val["eligible"] & val["history_ok"]).sum())
    assert settings["validation_checked_days"] == checked
    assert settings["validation_ml_target"] == round(settings["ml_target_alert_rate"] * checked)
    assert int(val["ml_flag"].sum()) == settings["validation_ml_target"]


@needs_outputs
def test_the_detector_trains_on_a_full_seasonal_cycle_when_earlier_history_exists(scored):
    _, frame, out, _, settings, plan = scored
    dev = out[(out["period"] == "development") & out["eligible"] & out["history_ok"]]
    if not (OUT / "validation" / "every_morning_forecast_extended.parquet").exists():
        pytest.skip("run run_pipeline.py to build the extended same-day history")
    assert plan[0][1] == str(frame["date"].min().date()) < "2026-01-01"
    assert (dev["date"].max() - dev["date"].min()).days >= 330        # about a year: every season once
    assert settings["trained_rows"] == len(dev)


@needs_outputs
def test_no_infinite_percentages(scored):
    _, _, out, _, _, _ = scored
    assert np.isfinite(alert_table(out)["pct_of_expected"].dropna()).all()


@needs_outputs
def test_switch_on_after_months_at_zero_is_caught(scored):
    _, _, out, _, _, _ = scored
    alerts = alert_table(out)
    zn16 = alerts[(alerts["meter_label"] == "ZN.16-PB_child__1_ch3pm")
                  & (alerts["date"].dt.strftime("%Y-%m") == "2026-06")]
    if zn16.empty and not (out["date"] >= "2026-06-01").any():
        pytest.skip("June is not in the data")
    assert len(zn16) > 0 and (zn16["direction"] == "higher").all()


@needs_outputs
def test_less_frequent_readings_are_not_missing_hours(scored):
    _, _, out, _, _, _ = scored
    q = quality_alerts(out)
    slow = q[q["title"] == "Readings less frequent than normal"]
    assert len(slow) > 0 and (slow["missing_hours"] == 0).all()


@needs_outputs
def test_evaluation_survives_periods_with_no_data(scored):
    """Partial data (for example no final test period yet) must not crash the evaluation."""
    daily, _, out, forest, settings, plan = scored
    ev = evaluate(out, quality_alerts(out), forest, settings, historical_register_check(daily, *KNOWN_REGISTER_JUMP), plan)
    assert {"periods", "final_test_agreement", "known_events", "synthetic_check"} <= set(ev)
    for s in ev["synthetic_check"].values():
        assert 0 <= s["either"] <= 100


@needs_outputs
def test_quality_episodes_and_historical_register_check(scored):
    daily, _, out, _, _, _ = scored
    q = quality_alerts(out)
    assert {"meter", "site"} <= set(q["scope"]) or set(q["scope"]) == {"meter"}
    assert (q["end"] >= q["start"]).all()
    reg = historical_register_check(daily, *KNOWN_REGISTER_JUMP)
    assert reg["flagged_as_data_problem"] and reg["reason"] == "register problem"
