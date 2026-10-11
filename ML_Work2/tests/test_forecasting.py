"""Checks that the forecast is honest and the numbers are computed correctly.

Run:  python -m pytest tests -q
Tests that need saved results are skipped until run_pipeline.py has run.
"""
import os
from datetime import date

import numpy as np
import pandas as pd
import pytest

from errors import Errors
from models.forecasting.features import Panel, origin_rows, training_rows
from models.forecasting.forecaster import Forecaster, linear_drivers, robust_linear
from tests.conftest import needs_data
from utilities.billing_rules import bill, cycles, service_class
from utilities.saudi_calendar import calendar_features
from utilities.scoring import accuracy


@needs_data
def test_forecast_inputs_never_see_the_future(inputs, forecast_end):
    """Scramble everything from the issue day onward: the inputs must not move."""
    daily, weather = inputs
    issue = pd.Timestamp("2026-05-17")
    clean = Panel(daily, weather, forecast_end)
    rng = np.random.default_rng(0)
    future = daily["date"] >= issue
    scrambled = daily.copy()
    scrambled.loc[future, "spread_kwh"] = rng.uniform(0, 5000, future.sum())
    scrambled.loc[future, "gold_kwh"] = rng.uniform(0, 5000, future.sum())
    scrambled.loc[future, "measurable_day"] = True
    hot = weather.copy()
    hot.loc[hot["date"] >= issue, "t_mean"] = 55.0
    dirty = Panel(scrambled, hot, forecast_end)
    t = clean.index_of(issue)
    a = origin_rows(clean, t, horizons=31).drop(columns=["actual_kwh", "target"])
    b = origin_rows(dirty, t, horizons=31).drop(columns=["actual_kwh", "target"])
    pd.testing.assert_frame_equal(a, b)


@needs_data
def test_training_only_uses_days_before_the_cutoff(inputs, forecast_end):
    daily, weather = inputs
    panel = Panel(daily, weather, forecast_end)
    cutoff = panel.index_of("2026-03-01")
    rows = training_rows(panel, range(8, cutoff, 5), cutoff, 0, 31)
    assert rows["target_idx"].max() < cutoff
    assert rows["origin_idx"].max() < cutoff


@needs_data
def test_measurable_gold_days_match_the_training_series(inputs):
    """On measurable days the gold-booked kWh and the outage-spread kWh agree."""
    daily, _ = inputs
    m = daily[daily["measurable_day"]]
    diff = (m["gold_kwh"] - m["spread_kwh"]).abs()
    assert (diff <= 0.05 * m["gold_kwh"].clip(lower=1) + 1.0).mean() > 0.99


@needs_data
def test_daily_totals_preserve_energy(inputs):
    daily, _ = inputs
    assert daily["gold_kwh"].sum() == pytest.approx(daily["spread_kwh"].sum(), rel=1e-9)


@needs_data
def test_every_meter_has_a_zone(inputs):
    daily, _ = inputs
    assert daily["zone"].notna().all() and (daily["zone"] > 0).all()


def test_accuracy_is_100_minus_wape():
    frame = pd.DataFrame({"actual": [100.0, 50.0], "abs_error": [10.0, 5.0]})
    assert accuracy(frame) == pytest.approx(90.0)


def test_robust_linear_resists_outliers():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3000, 3))
    beta = np.array([0.8, -0.3, 0.5])
    y = x @ beta + rng.normal(0, 0.05, 3000)
    bad = rng.random(3000) < 0.05
    y[bad] += rng.normal(0, 3, bad.sum())
    ols = np.linalg.lstsq(x, y, rcond=None)[0]
    assert np.abs(robust_linear(x, y) - beta).max() < np.abs(ols - beta).max()


def test_forecast_beyond_trained_bands_raises_instead_of_nan():
    zero = ["seasonal_effect_change", "weather_anomaly_effect", "dow_ratio", "trend_7_vs_28",
            "yesterday_vs_base", "last_3_days_vs_base", "base_ramadan_share", "base_off_share"]
    rows = pd.DataFrame({c: [0.0, 0.0] for c in zero})
    rows["days_ahead"] = [3, 9]
    rows["ramadan"], rows["eid"], rows["national_holiday"], rows["log_base"] = 0, False, False, 1.0
    model = Forecaster()
    model.parts = [(0, 5, pd.Series(0.0, index=linear_drivers(rows).columns), None)]
    with pytest.raises(Errors.Misc.HorizonExceeded, match="outside the trained bands"):
        model.predict_kwh(rows)


def test_saudi_calendar():
    cal = calendar_features(pd.Series(pd.to_datetime(["2026-02-20", "2026-02-21", "2026-02-22",
                                                      "2026-03-21", "2026-05-27", "2026-09-23"])))
    assert cal["weekend"].tolist() == [1, 1, 0, 1, 0, 0]          # Friday, Saturday weekend
    assert cal["ramadan"].tolist()[:3] == [1, 1, 1]
    assert cal["eid"].tolist()[3:5] == [1, 1]
    assert cal["national_holiday"].tolist()[2] == 1 and cal["national_holiday"].tolist()[5] == 1


def test_bill_rules():
    # The facility's own worked example: 10,580 kWh on a 100 A breaker.
    b = bill(10580, service_class(100)[1])
    assert (b["kwh_at_022"], b["kwh_at_032"]) == (6000, 4580)
    assert b["bill_sar"] == pytest.approx(3220.69)
    small = bill(5000, 10.0)
    assert small["energy_032_sar"] == 0 and small["bill_sar"] == pytest.approx((5000 * 0.22 + 10) * 1.15)
    for x in (0, 5999.99, 6000, 6000.01, 12345.6):
        p = bill(x, 21.0)
        assert p["energy_022_sar"] + p["energy_032_sar"] + p["service_fee_sar"] + p["vat_sar"] == pytest.approx(p["bill_sar"])
    assert [service_class(a)[1] for a in (80, 100, 199, 200, 300, 400, 401)] == [10, 15, 15, 21, 22, 25, 30]


def test_billing_cycles_start_on_the_first_of_the_year():
    c = cycles(date(2026, 1, 1), date(2026, 12, 31))
    assert [(str(x["start"]), str(x["end"])) for x in c[:3]] == [
        ("2026-01-01", "2026-01-30"), ("2026-01-31", "2026-03-01"), ("2026-03-02", "2026-03-31")]
    assert sum(x["full"] for x in c) == 12
    assert str(cycles(date(2027, 1, 1), date(2027, 12, 31))[0]["start"]) == "2027-01-01"


@pytest.mark.skipif(not (os.environ.get("SADINHUB_PASSWORD") or os.environ.get("PGPASSWORD")),
                    reason="database password not set")
def test_consumption_rule_reproduces_gold_exactly():
    from config.settings import load_settings
    from main.app import build_service_container
    from utilities.gold_rule import apply_gold_rule
    services = build_service_container(load_settings())
    raw = services.data.load_database_readings()
    validated = apply_gold_rule(raw[~raw["meter_label"].str.startswith("ZN.13-")])
    try:
        result = services.data.check_rule_against_gold(validated)
    except Errors.HTTP.InternalServerError:
        pytest.skip("gold.energy_consumption_validated is not in this database")
    assert result["meter_days_exact_within_0.01_kwh"] == result["meter_days_compared"]
    assert result["our_total_kwh_same_meter_days"] == pytest.approx(2020697.5, abs=0.1)


def test_derived_holidays_match_the_published_umm_al_qura_dates():
    """The calendar is derived, not typed in: it must reproduce the dates the project used for 2025-2026."""
    from utilities.saudi_calendar import islamic_ranges
    r = islamic_ranges(2025, 2026)
    assert [x for x in r["ramadan"] if x[0][:4] in ("2025", "2026")] == [
        ("2025-03-01", "2025-03-29"), ("2026-02-18", "2026-03-19")]
    assert [x for x in r["fitr"] if x[0][:4] in ("2025", "2026")] == [
        ("2025-03-30", "2025-04-02"), ("2026-03-20", "2026-03-23")]
    assert [x for x in r["adha"] if x[0][:4] in ("2025", "2026")] == [
        ("2025-06-05", "2025-06-09"), ("2026-05-26", "2026-05-30")]


def test_calendar_covers_any_year_without_editing_a_table():
    cal = calendar_features(pd.Series(pd.to_datetime(["2028-02-01", "2028-06-01"])))
    assert cal["ramadan"].tolist() == [1, 0]
