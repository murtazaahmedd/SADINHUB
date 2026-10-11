"""Checks that the forecast is honest and the numbers are computed correctly.

Run:  .venv\\Scripts\\python.exe -m pytest tests -q
Tests that need the database are skipped when no password is set.
"""
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forecasting import config  # noqa: E402
from forecasting.features import Panel, origin_rows, training_rows  # noqa: E402
from forecasting.saudi_calendar import calendar_features  # noqa: E402
from forecasting.scoring import accuracy  # noqa: E402

DAILY = config.DATA_DIR / "daily_actuals.parquet"
WEATHER = config.DATA_DIR / "weather_riyadh_daily.csv"
needs_data = pytest.mark.skipif(not (DAILY.exists() and WEATHER.exists()),
                                reason="run run_pipeline.py first")


def _inputs():
    return pd.read_parquet(DAILY), pd.read_csv(WEATHER, parse_dates=["date"])


@needs_data
def test_forecast_inputs_never_see_the_future():
    """Scramble everything from the issue day onward: the inputs must not move."""
    daily, weather = _inputs()
    issue = pd.Timestamp("2026-05-17")
    clean = Panel(daily, weather, config.FORECAST_END)

    rng = np.random.default_rng(0)
    future = daily["date"] >= issue
    scrambled = daily.copy()
    scrambled.loc[future, "spread_kwh"] = rng.uniform(0, 5000, future.sum())
    scrambled.loc[future, "gold_kwh"] = rng.uniform(0, 5000, future.sum())
    scrambled.loc[future, "measurable_day"] = True
    hot = weather.copy()
    hot.loc[hot["date"] >= issue, "t_mean"] = 55.0
    dirty = Panel(scrambled, hot, config.FORECAST_END)

    t = clean.index_of(issue)
    a = origin_rows(clean, t, horizons=31).drop(columns=["actual_kwh", "target"])
    b = origin_rows(dirty, t, horizons=31).drop(columns=["actual_kwh", "target"])
    pd.testing.assert_frame_equal(a, b)


@needs_data
def test_training_only_uses_days_before_the_cutoff():
    daily, weather = _inputs()
    panel = Panel(daily, weather, config.FORECAST_END)
    cutoff = panel.index_of("2026-03-01")
    rows = training_rows(panel, range(8, cutoff, 5), cutoff, 0, 31)
    assert rows["target_idx"].max() < cutoff
    assert rows["origin_idx"].max() < cutoff


@needs_data
def test_measurable_gold_days_match_the_training_series():
    """On measurable days the gold-booked kWh and the outage-spread kWh agree."""
    daily, _ = _inputs()
    m = daily[daily["measurable_day"]]
    diff = (m["gold_kwh"] - m["spread_kwh"]).abs()
    assert (diff <= 0.05 * m["gold_kwh"].clip(lower=1) + 1.0).mean() > 0.99


@needs_data
def test_daily_totals_preserve_energy():
    daily, _ = _inputs()
    assert daily["gold_kwh"].sum() == pytest.approx(daily["spread_kwh"].sum(), rel=1e-9)


def test_accuracy_is_100_minus_wape():
    frame = pd.DataFrame({"actual": [100.0, 50.0], "abs_error": [10.0, 5.0]})
    assert accuracy(frame) == pytest.approx(90.0)


def test_saudi_calendar():
    cal = calendar_features(pd.Series(pd.to_datetime(["2026-02-20", "2026-02-21", "2026-02-22",
                                                      "2026-03-21", "2026-05-27", "2026-09-23"])))
    assert cal["weekend"].tolist() == [1, 1, 0, 1, 0, 0]          # Friday, Saturday weekend
    assert cal["ramadan"].tolist()[:3] == [1, 1, 1]
    assert cal["eid"].tolist()[3:5] == [1, 1]
    assert cal["national_holiday"].tolist()[2] == 1 and cal["national_holiday"].tolist()[5] == 1


def test_bill_rules():
    from forecasting.billing import bill, cycles, service_class
    # The facility's own worked example: 10,580 kWh on a 100 A breaker.
    b = bill(10580, service_class(100)[1])
    assert (b["kwh_at_022"], b["kwh_at_032"]) == (6000, 4580)
    assert b["bill_sar"] == pytest.approx(3220.69)
    # Below the 6,000 kWh step only 0.22 applies; parts always add up to the bill.
    small = bill(5000, 10.0)
    assert small["energy_032_sar"] == 0 and small["bill_sar"] == pytest.approx((5000 * 0.22 + 10) * 1.15)
    for x in (0, 5999.99, 6000, 6000.01, 12345.6):
        p = bill(x, 21.0)
        assert p["energy_022_sar"] + p["energy_032_sar"] + p["service_fee_sar"] + p["vat_sar"] == pytest.approx(p["bill_sar"])
    # Breaker classes: boundaries exactly as in the tariff table.
    assert [service_class(a)[1] for a in (80, 100, 199, 200, 300, 400, 401)] == [10, 15, 15, 21, 22, 25, 30]
    # 30-day cycles from 1 January: 1-30 Jan, 31 Jan-1 Mar, 2-31 Mar ...
    c = cycles()
    assert [(str(x["start"]), str(x["end"])) for x in c[:3]] == [
        ("2026-01-01", "2026-01-30"), ("2026-01-31", "2026-03-01"), ("2026-03-02", "2026-03-31")]
    assert sum(x["full"] for x in c) == 12


@pytest.mark.skipif(not (os.environ.get("SADINHUB_PASSWORD") or os.environ.get("PGPASSWORD")),
                    reason="database password not set")
def test_consumption_rule_reproduces_gold_exactly():
    from forecasting import readings
    raw = readings.load_database_readings()
    validated = readings.apply_gold_rule(raw[~raw["meter_label"].str.startswith("ZN.13-")])
    result = readings.check_rule_against_gold(validated)
    assert result["meter_days_exact_within_0.01_kwh"] == result["meter_days_compared"]
    assert result["our_total_kwh_same_meter_days"] == pytest.approx(2020697.5, abs=0.1)
