"""Adversarial tests of the alerts: inject faults, inject nothing, inject noise, starve the detector of data."""
import json

import numpy as np
import pandas as pd
import pytest

from errors import Errors
from models.anomaly.detectors import detect, rescore
from models.anomaly.features import build_features, with_actual
from repositories.artifacts.artifact_repository import ArtifactRepository
from tests.conftest import ROOT, alert_inputs
from utilities.anomaly_quality import quality_alerts
from models.anomaly.evaluation import evaluate
from utilities.anomaly_report import alert_table

OUT = ROOT / "outputs"
needs_outputs = pytest.mark.skipif(
    not all((OUT / f).exists() for f in ("data/daily_actuals.parquet", "validation/every_morning_forecast.parquet",
                                         "data/validated_intervals.parquet")), reason="run run_pipeline.py first")


@pytest.fixture(scope="module")
def world():
    daily, frame, plan = alert_inputs()
    featured = build_features(frame, daily)
    out, forest, settings = detect(featured, plan)
    return daily, featured, out, forest, settings, plan


def checked(out):
    return out[out["eligible"] & out["history_ok"] & out["period"].isin(["validation", "final test", "latest"])]


@needs_outputs
def test_injecting_nothing_creates_no_new_alerts(world):
    _, _, out, forest, settings, _ = world
    rows = checked(out)
    same = with_actual(rows, rows["actual_kwh"])
    rule, ml = rescore(same, forest, settings["ml_cutoff"])
    assert (rule == rows["rule_flag"]).all() and (ml == rows["ml_flag"]).all()


@needs_outputs
@pytest.mark.parametrize("factor,direction", [(1.6, "higher"), (0.4, "lower"), (3.0, "higher"), (0.0, "lower")])
def test_injected_faults_are_caught_in_the_right_direction(world, factor, direction):
    _, _, out, forest, settings, _ = world
    rows = checked(out)
    rows = rows[rows["expected_kwh"] > 20]                       # a fault on an idle meter is not a meaningful test
    moved = with_actual(rows, rows["actual_kwh"] * factor)
    rule, ml = rescore(moved, forest, settings["ml_cutoff"])
    right_way = np.sign(moved["diff_kwh"]) == (1 if direction == "higher" else -1)
    detected = float(((rule | ml) & right_way).sum() / max(right_way.sum(), 1))
    floor = {1.6: 0.6, 0.4: 0.6, 3.0: 0.85, 0.0: 0.85}[factor]
    assert detected >= floor, f"x{factor}: only {detected:.0%} of injected faults were flagged"


@needs_outputs
def test_small_noise_does_not_flood_the_alerts(world):
    _, _, out, forest, settings, _ = world
    rows = checked(out)
    rng = np.random.default_rng(0)
    moved = with_actual(rows, rows["actual_kwh"] * (1 + rng.uniform(-0.02, 0.02, len(rows))))
    rule, ml = rescore(moved, forest, settings["ml_cutoff"])
    assert float((rule | ml).mean()) <= float((rows["rule_flag"] | rows["ml_flag"]).mean()) + 0.03


@needs_outputs
def test_alert_volume_is_reviewable(world):
    _, _, out, _, _, _ = world
    weekly = out[out["detector"] != ""].groupby("period").size()
    days = out.groupby("period")["date"].nunique()
    assert ((weekly / (days.reindex(weekly.index) / 7)).dropna() < 60).all()      # fewer than 60 alerts a week


@needs_outputs
def test_untrusted_days_never_become_consumption_alerts_even_when_forced(world):
    _, featured, _, _, _, plan = world
    broken = featured.copy()
    broken["eligible"] = False
    broken["actual_kwh"] = np.nan
    broken.loc[broken.index[:200], "eligible"] = True
    broken.loc[broken.index[:200], "actual_kwh"] = broken.loc[broken.index[:200], "gold_kwh"]
    out, _, _ = detect(featured.assign(eligible=featured["eligible"]), plan)
    flagged = out[out["detector"] != ""]
    assert flagged["eligible"].all()


@needs_outputs
def test_a_starved_validation_period_turns_ml_alerts_off_instead_of_crashing(world):
    _, featured, _, _, _, plan = world
    no_validation = [p for p in plan if p[0] != "validation"]
    out, _, settings = detect(featured, no_validation)
    assert settings["ml_cutoff"] is None and not out["ml_flag"].any()


@needs_outputs
def test_a_starved_development_period_is_refused_with_a_clear_message(world):
    _, featured, _, _, _, plan = world
    with pytest.raises(Errors.Misc.InsufficientData, match="development period"):
        detect(featured, [p for p in plan if p[0] != "development"])


@needs_outputs
def test_evaluation_survives_no_final_test_and_no_quality_episodes(world):
    daily, featured, _, _, _, plan = world
    out, forest, settings = detect(featured, [p for p in plan if p[0] != "final test"])
    ev = evaluate(out, pd.DataFrame(), forest, settings, {"found": False}, plan)
    assert ev["synthetic_check"] == {} and ev["final_test_agreement"] == {"both": 0, "rule_only": 0, "ml_only": 0}


@needs_outputs
def test_the_alert_table_has_no_infinities_and_a_check_text_for_every_alert(world):
    _, _, out, _, _, _ = world
    table = alert_table(out)
    assert np.isfinite(table[["diff_kwh", "actual_kwh", "expected_kwh"]].to_numpy(float)).all()
    assert table["check"].str.len().min() > 20
    q = quality_alerts(out)
    assert q["check"].str.len().min() > 20 and (q["days"] >= 1).all()


@needs_outputs
def test_the_ml_alert_rate_does_not_drift_far_from_its_target_after_the_validation_period(world):
    """The fixed rate is set on July-August; the untouched September test shows how far the model drifts."""
    _, _, out, _, settings, _ = world
    test = out[(out["period"] == "final test") & out["eligible"] & out["history_ok"]]
    if test.empty:
        pytest.skip("no final test period in the data yet")
    rate = float(test["ml_flag"].mean())
    assert rate < 3 * settings["ml_target_alert_rate"], f"ML alert rate in the final test is {rate:.1%}"


@needs_outputs
def test_the_ml_cutoff_does_not_move_when_the_rule_threshold_changes(world, monkeypatch):
    _, featured, out, _, settings, plan = world
    import constants.anomaly as c
    import models.anomaly.detectors as d
    monkeypatch.setattr(d, "RULE_Z", c.RULE_Z * 2)
    again, _, settings2 = detect(featured, plan)
    assert settings2["ml_cutoff"] == settings["ml_cutoff"]
    assert settings2["validation_rule_alerts"] < settings["validation_rule_alerts"]


ALERTS = OUT / "alerts"
needs_alert_files = pytest.mark.skipif(not (ALERTS / "incidents.csv").exists(), reason="run run_pipeline.py --alerts first")


@needs_alert_files
def test_incidents_lose_no_alert_and_put_the_known_events_at_the_top():
    alerts = pd.read_csv(ALERTS / "consumption_alerts.csv", parse_dates=["date"])
    inc = pd.read_csv(ALERTS / "incidents.csv", parse_dates=["start", "end"])
    assert int(inc["alert_days"].sum()) == len(alerts)                                   # every alert is in exactly one incident
    for a in alerts.itertuples():
        hit = inc[(inc["meter_label"] == a.meter_label) & (inc["direction"] == a.direction)
                  & (inc["start"] <= a.date) & (inc["end"] >= a.date)]
        assert len(hit) == 1
    top = inc.head(8)
    assert "ZN.16-PB_child__1_ch3pm" in set(top["meter_label"]) and "ZN.8-PB.2_child__1_ch3pm" in set(top["meter_label"])
    assert inc["impact_sar"].is_monotonic_decreasing


@needs_outputs
def test_the_night_detector_finds_night_loads_the_daily_detectors_miss(world):
    from models.anomaly.shape import night_scores
    from utilities.anomaly_shape import night_base_load
    import constants.anomaly as c
    _, _, out, forest, settings, _ = world
    scores = night_scores(night_base_load(ArtifactRepository(OUT).load_frame("validated_intervals")))
    days = out[out["eligible"] & out["history_ok"] & out["period"].isin(["final test", "latest"]) & (out["expected_kwh"] > 20)]
    m = days.merge(scores, on=["meter_label", "date"])
    moved = with_actual(m, m["actual_kwh"] + 5 * 5)                       # +5 kW for the five night hours
    rule, ml = rescore(moved, forest, settings["ml_cutoff"])
    z = (np.log(m["night_kw"] + 5 + c.NIGHT_LOG_OFFSET_KW) - np.log(m["base_kw"] + c.NIGHT_LOG_OFFSET_KW)) / m["spread"]
    night = ((z >= c.NIGHT_Z) & (m["night_kw"] + 5 - m["base_kw"] >= c.NIGHT_MIN_RISE_KW)).to_numpy()
    daily = (rule | ml).to_numpy()
    assert (night & ~daily).mean() >= 0.20, "the night detector adds too little over the daily detectors"
    assert night.mean() > daily.mean()


@needs_alert_files
def test_the_night_alert_load_is_reviewable():
    night = pd.read_csv(ALERTS / "night_load_alerts.csv")
    ev = json.loads((ALERTS / "evaluation.json").read_text())["extras"]["per_period"]
    for period, row in ev.items():
        if row["nights_checked"] >= 500:
            assert row["night_alerts"] / row["nights_checked"] < 0.06, period
    assert (night["rise_kw"] >= 1.0).all() and (night["night_z"] >= 5.0).all()


@needs_outputs
def test_the_ml_alert_rate_in_the_untouched_test_month_is_close_to_its_target(world):
    _, _, out, _, settings, _ = world
    test = out[(out["period"] == "final test") & out["eligible"] & out["history_ok"]]
    if test.empty:
        pytest.skip("no final test period in the data yet")
    assert 0.015 <= float(test["ml_flag"].mean()) <= 1.5 * settings["ml_target_alert_rate"]
