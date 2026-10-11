"""Tests of incidents and night base load, on synthetic data with known answers."""
import numpy as np
import pandas as pd
import pytest

from constants.tariff import RATE_UP_TO_LIMIT
from models.anomaly.shape import night_alerts, night_scores
from utilities.anomaly_incidents import build_incidents
from utilities.anomaly_shape import night_base_load


def alert_rows(meter, days, direction="higher", diff=50.0, zone=1):
    return pd.DataFrame({"meter_label": meter, "zone": zone, "date": pd.to_datetime(days), "period": "validation",
                         "direction": direction, "detector": "both", "diff_kwh": diff, "z": 4.0, "check": "check it"})


# ---- incidents ---------------------------------------------------------------------------

def test_consecutive_alert_days_become_one_incident_with_its_impact():
    inc = build_incidents(alert_rows("M1", ["2026-05-01", "2026-05-02", "2026-05-04"]), pd.Timestamp("2026-09-01"))
    assert len(inc) == 1                                                  # one quiet day (May 3) does not split it
    r = inc.iloc[0]
    assert (r.alert_days, r.span_days, r.total_diff_kwh) == (3, 4, 150.0)
    assert r.impact_sar == pytest.approx(150 * RATE_UP_TO_LIMIT, abs=0.01)


def test_a_longer_gap_or_a_different_direction_starts_a_new_incident():
    a = pd.concat([alert_rows("M1", ["2026-05-01", "2026-05-05"]), alert_rows("M1", ["2026-05-01"], "lower", -20.0)])
    inc = build_incidents(a, pd.Timestamp("2026-09-01"))
    assert len(inc) == 3 and set(inc["direction"]) == {"higher", "lower"}


def test_incidents_are_ranked_by_money_and_the_active_flag_follows_the_data_end():
    a = pd.concat([alert_rows("SMALL", ["2026-05-01"], diff=10.0), alert_rows("BIG", ["2026-08-30", "2026-08-31"], diff=500.0)])
    inc = build_incidents(a, pd.Timestamp("2026-09-01"))
    assert inc["meter_label"].tolist() == ["BIG", "SMALL"] and inc["active"].tolist() == [True, False]


def test_no_alerts_gives_an_empty_incident_table_with_the_right_columns():
    inc = build_incidents(alert_rows("M1", [])[0:0], pd.Timestamp("2026-09-01"))
    assert inc.empty and "impact_sar" in inc.columns


# ---- night base load ---------------------------------------------------------------------

def five_minute_intervals(meter="M1", days=60, night_kw=2.0, day_kw=10.0, bump_day=None, bump_kw=0.0):
    ts = pd.date_range("2026-03-01", periods=days * 288, freq="5min")
    kw = np.where(ts.hour < 5, night_kw, day_kw).astype(float)
    if bump_day is not None:
        kw = np.where((ts.normalize() == pd.Timestamp(bump_day)) & (ts.hour < 5), night_kw + bump_kw, kw)
    return pd.DataFrame({"meter_label": meter, "saudi": ts, "date": ts.normalize(), "consumption_kwh": kw * 5 / 60,
                         "gap_minutes": 5.0, "interval_quality": "NORMAL_INTERVAL", "status": "KEEP"})


def test_night_base_load_measures_only_the_night_hours():
    night = night_base_load(five_minute_intervals(night_kw=2.0, day_kw=50.0))
    assert night["night_kw"].round(6).eq(2.0).all() and night["night_hours"].round(6).eq(5.0).all()


def test_nights_with_too_little_normal_data_are_not_trusted():
    iv = five_minute_intervals(days=3)
    iv.loc[(iv["date"] == "2026-03-02") & (iv["saudi"].dt.hour < 4), "interval_quality"] = "LONG_GAP"   # only 1 hour left
    iv.loc[(iv["date"] == "2026-03-03") & (iv["saudi"].dt.hour < 5), "status"] = "REGISTER_JUMP"
    night = night_base_load(iv)
    assert set(night["date"]) == {pd.Timestamp("2026-03-01")}


def test_a_night_load_that_the_daily_total_would_hide_is_flagged():
    iv = five_minute_intervals(bump_day="2026-04-15", bump_kw=5.0)        # +5 kW for 5 hours = 25 kWh in a 290 kWh day
    alerts = night_alerts(night_scores(night_base_load(iv)))
    assert alerts["date"].tolist() == [pd.Timestamp("2026-04-15")]
    assert alerts["rise_kw"].iloc[0] == pytest.approx(5.0, abs=0.01)


def test_a_steady_meter_and_a_new_meter_with_no_history_raise_no_night_alerts():
    steady = night_alerts(night_scores(night_base_load(five_minute_intervals())))
    new = night_alerts(night_scores(night_base_load(five_minute_intervals(days=10, bump_day="2026-03-09", bump_kw=9.0))))
    assert steady.empty and new.empty                                     # fewer than 14 earlier nights: nothing to compare with


def test_a_night_is_judged_only_against_earlier_nights():
    iv = five_minute_intervals(bump_day="2026-04-15", bump_kw=5.0)
    future_changed = five_minute_intervals(bump_day="2026-04-15", bump_kw=5.0)
    later = future_changed["date"] > "2026-04-20"
    future_changed.loc[later & (future_changed["saudi"].dt.hour < 5), "consumption_kwh"] *= 40
    a, b = night_scores(night_base_load(iv)), night_scores(night_base_load(future_changed))
    pd.testing.assert_frame_equal(a[a["date"] <= "2026-04-20"].reset_index(drop=True),
                                  b[b["date"] <= "2026-04-20"].reset_index(drop=True))
