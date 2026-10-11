"""Tests against real, independent ground truth: the 29 utility bills the facility recorded (May 14 - Jun 12, 2025).

Each bill gives the energy the grid billed and the building system's own figure for the same period. They need the
warehouse (SADINHUB_PASSWORD) and the saved results.
"""
import os

import pandas as pd
import pytest
from sqlalchemy import create_engine, text

from config.settings import load_settings
from database.session import READ_ONLY_OPTIONS
from tests.conftest import ROOT

pytestmark = pytest.mark.database
OUT = ROOT / "outputs"
needs_results = pytest.mark.skipif(not (OUT / "data" / "daily_actuals.parquet").exists(), reason="run run_pipeline.py first")

BILLS = """SELECT em.meter_label, b.start_period, b.end_period, b.consumption_reading AS grid_kwh,
                  b.system_consumption_reading AS bms_kwh
           FROM bronze.energy_bills_tracking b
           JOIN bronze.energy_meters em ON em.grid_billing_account_id = b.grid_billing_account_id
           JOIN bronze.device d ON d.id = em.device_id
           WHERE NOT (em.meter_label = 'ZN.20-PB.2_child__1_ch3pm' AND d.name = 'ZN.20-PB.1_child__1_ch3pm')"""


@pytest.fixture(scope="module")
def comparison():
    if not (os.environ.get("SADINHUB_PASSWORD") or os.environ.get("PGPASSWORD")):
        pytest.skip("database password not set")
    engine = create_engine(load_settings().database_url, connect_args={"options": READ_ONLY_OPTIONS})
    with engine.connect() as conn:
        bills = pd.read_sql(text(BILLS), conn)
    daily = pd.read_parquet(OUT / "data" / "daily_actuals.parquet")
    iv = pd.read_parquet(OUT / "data" / "validated_intervals.parquet", columns=["meter_label", "date", "consumption_kwh"])
    tz = pd.Timedelta(hours=3)
    rows = []
    for r in bills.itertuples():
        s = (r.start_period.tz_convert("UTC").tz_localize(None) + tz).normalize()
        e = (r.end_period.tz_convert("UTC").tz_localize(None) + tz).normalize()
        d = daily[(daily["meter_label"] == r.meter_label) & (daily["date"] >= s) & (daily["date"] <= e)]
        i = iv[(iv["meter_label"] == r.meter_label) & (iv["date"] >= s) & (iv["date"] <= e)]
        rows.append({"meter": r.meter_label, "start": s, "end": e, "grid": r.grid_kwh, "bms": r.bms_kwh,
                     "gold": i["consumption_kwh"].sum(), "spread": d["spread_kwh"].sum(),
                     "untrusted_days": int((~d["measurable_day"]).sum())})
    return pd.DataFrame(rows)


@needs_results
def test_where_every_day_is_trusted_our_energy_equals_the_building_systems_own_figure(comparison):
    full = comparison[comparison["untrusted_days"] == 0]
    gap = (100 * (full["gold"] / full["bms"] - 1)).abs()
    assert len(full) >= 15 and (gap <= 0.5).sum() >= 15, gap.sort_values().tail(5)


@needs_results
def test_time_spread_booking_matches_the_utility_bill_for_meters_with_a_short_outage(comparison):
    """The six meters with a ~3.6 day outage: spread booking is within 3% of the grid, gold booking is 29-48% above it."""
    short = comparison[(comparison["untrusted_days"] == 4) & ((comparison["bms"] / comparison["grid"] - 1).abs() < 0.15)]
    assert len(short) >= 6                      # ZN.8-PB.2 is excluded: its building-system reading is 91% below the bill
    spread_err = (100 * (short["spread"] / short["grid"] - 1)).abs()
    gold_err = (100 * (short["gold"] / short["grid"] - 1)).abs()
    assert (spread_err <= 3.5).all(), spread_err.max()
    assert (gold_err >= 25).all(), gold_err.min()


@needs_results
def test_every_meter_with_missing_days_in_a_bill_window_has_a_data_problem_episode(comparison):
    q = pd.read_csv(OUT / "alerts" / "data_problem_alerts.csv", parse_dates=["start", "end"])
    gappy = comparison[comparison["untrusted_days"] > 0]
    assert len(gappy) >= 9
    for r in gappy.itertuples():
        hit = q[((q["meter_label"] == r.meter) | (q["scope"] == "site")) & (q["start"] <= r.end) & (q["end"] >= r.start)]
        assert len(hit) >= 1, r.meter


@needs_results
def test_clean_meters_track_the_utility_within_a_few_percent_and_total_energy_is_close(comparison):
    clean = comparison[(comparison["untrusted_days"] == 0)]
    close = clean[(100 * (clean["spread"] / clean["grid"] - 1)).abs() <= 2.5]
    assert len(close) >= 10                                              # about 0.7% below the utility is typical
    assert (100 * (close["spread"].sum() / close["grid"].sum() - 1)) == pytest.approx(-0.6, abs=1.0)
