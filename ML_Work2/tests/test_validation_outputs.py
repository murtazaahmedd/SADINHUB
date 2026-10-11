"""End-to-end validation of the saved results: data integrity, forecast sanity, honest accuracy.

These run against the files the last pipeline run wrote, so they validate the real results
(and are skipped until a run exists).
"""
import json

import numpy as np
import pandas as pd
import pytest

from constants.data_rules import MAX_PLAUSIBLE_METER_KW
from repositories.artifacts.artifact_repository import ArtifactRepository
from tests.conftest import ROOT
from utilities.scoring import accuracy, all_levels, meter_periods

OUT = ROOT / "outputs"
store = ArtifactRepository(OUT)
needs_results = pytest.mark.skipif(not (OUT / "forecasts" / "forecast_info.json").exists(), reason="run run_pipeline.py first")


@pytest.fixture(scope="module")
def daily():
    return store.load_frame("daily_actuals")


@pytest.fixture(scope="module")
def last_complete(daily):
    return daily.loc[daily["complete_day"], "date"].max()


@pytest.fixture(scope="module")
def forecast():
    return store.load_frame("forecast_meters", parse_dates=["date"])


# ---- data integrity ---------------------------------------------------------------------

@needs_results
def test_every_meter_reports_through_the_last_day(daily, last_complete):
    assert daily["meter_label"].nunique() == 31
    assert (daily.groupby("meter_label")["date"].max() >= last_complete).all()
    assert daily["zone"].nunique() == 21 and daily["zone"].notna().all()


@needs_results
def test_the_excel_handover_is_continuous_and_physically_plausible():
    db, xl = store.load_frame("raw_readings_database"), store.load_frame("raw_readings_excel")
    assert xl["meter_label"].nunique() == 31, "an export is missing for some meter"
    last = db.sort_values("ts_utc").groupby("meter_label").agg(ts=("ts_utc", "last"), kwh=("kwh", "last"))
    first = xl.sort_values("ts_utc").groupby("meter_label").agg(ts=("ts_utc", "first"), kwh=("kwh", "first"))
    step = (first["kwh"] - last["kwh"]).reindex(last.index)
    hours = ((first["ts"] - last["ts"]).dt.total_seconds() / 3600).reindex(last.index)
    assert (step >= 0).all(), step[step < 0]                              # the register never goes backwards at the handover
    assert (step / hours.clip(lower=1 / 60) <= MAX_PLAUSIBLE_METER_KW).all()
    both = pd.concat([db[["meter_label", "ts_utc"]], xl[["meter_label", "ts_utc"]]])
    assert both.duplicated().sum() == 0, "readings present in both the database and the Excel exports"


@needs_results
def test_booked_energy_obeys_an_exact_accounting_identity_for_every_meter():
    """booked = net register rise + ignored decrements - ignored jumps, to the last decimal."""
    iv = store.load_frame("validated_intervals", columns=["meter_label", "kwh", "consumption_kwh", "status"])
    for label, g in iv.groupby("meter_label"):
        step = g["kwh"].diff()
        decrements = -step[g["status"] == "UNRESOLVED_DECREMENT"].sum()
        jumps = step[g["status"] == "REGISTER_JUMP"].sum()
        rise = g["kwh"].iloc[-1] - g["kwh"].iloc[0]
        assert g["consumption_kwh"].sum() == pytest.approx(rise + decrements - jumps, abs=1e-5), label


@needs_results
def test_a_register_reset_books_no_energy_and_is_flagged_as_a_data_problem():
    """ZN.13-PB.2's register fell from 3,723.5 to 778.5 kWh on 2025-02-05: that interval must book nothing."""
    iv = store.load_frame("validated_intervals", columns=["meter_label", "ts_utc", "kwh", "consumption_kwh", "status"])
    g = iv[(iv["meter_label"] == "ZN.13-PB.2_child__1_ch3pm") & (iv["ts_utc"].dt.strftime("%Y-%m-%d") == "2025-02-05")]
    reset = g[g["status"] == "UNRESOLVED_DECREMENT"]
    assert len(reset) == 1 and reset["consumption_kwh"].iloc[0] == 0.0 and (g["consumption_kwh"].dropna() >= 0).all()


@needs_results
def test_no_meter_day_exceeds_the_physical_maximum(daily):
    assert (daily["gold_kwh"].dropna() >= 0).all()
    ok = daily[daily["measurable_day"]]
    assert (ok["gold_kwh"] <= MAX_PLAUSIBLE_METER_KW * 24).all()


# ---- forecast sanity --------------------------------------------------------------------

@needs_results
def test_the_forecast_is_complete_finite_and_physically_bounded(forecast, last_complete):
    info = json.loads((OUT / "forecasts" / "forecast_info.json").read_text())
    end = pd.Timestamp(info["forecast_until"])
    days = pd.date_range(last_complete + pd.Timedelta(days=1), end)
    assert forecast["meter_label"].nunique() == 31
    assert set(forecast["date"].unique()) == set(days)
    assert len(forecast) == 31 * len(days)
    assert np.isfinite(forecast["forecast_kwh"]).all() and (forecast["forecast_kwh"] >= 0).all()
    assert (forecast["forecast_kwh"] <= MAX_PLAUSIBLE_METER_KW * 24).all()


@needs_results
def test_zone_and_facility_forecasts_are_exact_sums_of_the_meters(forecast):
    zones = store.load_frame("forecast_zones", parse_dates=["date"])
    fac = store.load_frame("forecast_facility", parse_dates=["date"])
    pd.testing.assert_series_equal(forecast.groupby("date")["forecast_kwh"].sum().round(6).reset_index(drop=True),
                                   fac["forecast_kwh"].round(6), check_names=False)
    assert zones["forecast_kwh"].sum() == pytest.approx(forecast["forecast_kwh"].sum())


@needs_results
def test_the_final_forecast_is_trained_on_all_available_data(daily, last_complete):
    info = json.loads((OUT / "forecasts" / "forecast_info.json").read_text())
    assert pd.Timestamp(info["model_trained_through"]) == last_complete
    assert pd.Timestamp(info["forecast_issued_for_morning_of"]) == last_complete + pd.Timedelta(days=1)
    assert info["training_examples"] > 1_000_000


@needs_results
def test_the_forecast_level_is_plausible_against_recent_actuals(daily, forecast, last_complete):
    recent = daily[daily["complete_day"] & (daily["date"] > last_complete - pd.Timedelta(days=28))]
    actual_mean = recent.groupby("date")["spread_kwh"].sum().mean()
    first_month = forecast[forecast["date"] <= forecast["date"].min() + pd.Timedelta(days=29)]
    forecast_mean = first_month.groupby("date")["forecast_kwh"].sum().mean()
    assert 0.5 < forecast_mean / actual_mean < 1.6, (forecast_mean, actual_mean)


# ---- honest accuracy --------------------------------------------------------------------

@pytest.fixture(scope="module")
def scoring_inputs():
    from models.forecasting.features import Panel
    daily = store.load_frame("daily_actuals")
    weather = store.load_frame("weather", parse_dates=["date"])
    last = daily.loc[daily["complete_day"], "date"].max()
    panel = Panel(daily, weather, pd.Timestamp(year=last.year, month=12, day=31))
    return panel, store.load_frame("every_morning_forecast")


@needs_results
def test_reported_accuracy_never_exceeds_100_and_matches_an_independent_recomputation(scoring_inputs, daily):
    summary = pd.DataFrame(store.load_json("accuracy_summary_json"))
    assert (summary["overall_accuracy"] <= 100).all()
    preds = scoring_inputs[1]
    d0 = preds[preds["days_ahead"] == 0].copy()
    d0["meter_label"] = np.array(scoring_inputs[0].meters)[d0["meter_idx"]]
    d0["date"] = scoring_inputs[0].dates[d0["target_idx"].to_numpy()]
    j = d0.merge(daily[["meter_label", "date", "gold_kwh", "measurable_day", "complete_day"]], on=["meter_label", "date"])
    j = j[j["measurable_day"] & j["complete_day"]]
    mine = 100 * (1 - (j["pred_kwh"] - j["gold_kwh"]).abs().sum() / j["gold_kwh"].sum())
    reported = summary.query("level == 'meter' and horizon == '1 day'")["overall_accuracy"].iloc[0]
    assert mine == pytest.approx(reported, abs=0.01)


@needs_results
@pytest.mark.parametrize("level,horizon", [("meter", "1d"), ("meter", "30d"), ("facility", "1d"), ("facility", "30d")])
def test_the_model_beats_a_naive_last_seven_days_baseline(scoring_inputs, level, horizon):
    panel, preds = scoring_inputs
    label = {"1d": 1, "30d": 30}[horizon]

    def score(p):
        per = meter_periods(panel, p, horizons=(label,))
        per["horizon"] = horizon
        rows = all_levels(per)
        return accuracy(rows[rows["level"] == level])

    naive = preds.copy()
    naive["pred_kwh"] = naive["base_kwh"]
    assert score(preds) > score(naive), f"{level} {horizon}: the model does not beat 'last 7 days held flat'"


@needs_results
def test_forecasts_are_not_systematically_biased_by_more_than_a_quarter(scoring_inputs):
    panel, preds = scoring_inputs
    per = meter_periods(panel, preds, horizons=(30,))
    m = per[per["measurable"]]
    ratio = m["forecast"].sum() / m["actual"].sum()
    assert 0.75 < ratio < 1.25, f"30-day forecast/actual = {ratio:.3f}"


# ---- bills and dashboard ----------------------------------------------------------------

@needs_results
def test_bills_add_up_and_are_never_negative():
    bills = store.load_frame("bill_forecast")
    for prefix in ("predicted",):
        parts = bills[[f"{prefix}_energy_022_sar", f"{prefix}_energy_032_sar", f"{prefix}_service_fee_sar",
                       f"{prefix}_vat_sar"]].sum(axis=1)
        assert (parts - bills[f"{prefix}_bill_sar"]).abs().max() < 0.02
    assert (bills["predicted_bill_sar"] >= 0).all() and (bills["predicted_kwh"] >= 0).all()
    assert set(bills["status"]) <= {"completed", "in progress", "upcoming", "runs into next year"}


@needs_results
def test_the_dashboard_is_built_complete_and_free_of_unfilled_placeholders():
    page = (ROOT / "dashboard" / "SADIN_Energy_Forecast_Dashboard.html").read_text(encoding="utf-8")
    assert "__YEAR__" not in page and "__NEXT_YEAR__" not in page and "/*__DATA__*/" not in page
    assert "/*__CHART_JS__*/" not in page and "Chart.js v" in page
    start = page.index("const D = ") + len("const D = ")
    data = json.JSONDecoder().raw_decode(page[start:])[0]
    assert len(data["units"]) == 1 + 21 + 31
    assert data["meta"]["year"] == pd.Timestamp(data["meta"]["issue_date"]).year
