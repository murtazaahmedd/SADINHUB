"""Adversarial tests of the forecasting model.

Part 1 uses small synthetic panels with known answers (constant, all-zero, short-history and
trailing-outage meters). Part 2 refits on perturbed real data (marked slow, needs saved results).
"""
import numpy as np
import pandas as pd
import pytest

from models.forecasting.features import Panel, origin_rows
from models.forecasting.forecaster import Forecaster
from repositories.artifacts.artifact_repository import ArtifactRepository
from services.forecast.forecast_service import ForecastService
from tests.conftest import needs_data

START = pd.Timestamp("2025-01-01")
DAYS = 420


def synthetic_weather():
    dates = pd.date_range("2010-01-01", START + pd.Timedelta(days=DAYS + 120), freq="D")
    doy = dates.dayofyear.to_numpy()
    mean = 26 - 12 * np.cos(2 * np.pi * (doy - 15) / 365.25)
    return pd.DataFrame({"date": dates, "t_mean": mean, "t_max": mean + 6, "t_min": mean - 6})


def meter_rows(label, kwh, zone=1, untrusted=None):
    dates = pd.date_range(START, periods=len(kwh), freq="D")
    kwh = np.asarray(kwh, float)
    bad = np.zeros(len(kwh), bool) if untrusted is None else untrusted
    return pd.DataFrame({"meter_label": label, "date": dates, "gold_kwh": kwh, "spread_kwh": kwh,
                         "measurable_day": ~bad, "complete_day": True, "outage_at_day_start": False, "zone": zone})


@pytest.fixture(scope="module")
def synthetic():
    rng = np.random.default_rng(5)
    t = np.arange(DAYS)
    weekly = 1 + 0.15 * np.isin(t % 7, [4, 5])
    meters = [
        meter_rows("ZN.1-PB_child__1_ch3pm", np.full(DAYS, 100.0)),                                    # constant
        meter_rows("ZN.2-PB_child__1_ch3pm", np.zeros(DAYS)),                                          # switched off for ever
        meter_rows("ZN.3-PB_child__1_ch3pm", 100 * weekly * (1 + 0.02 * rng.normal(size=DAYS))),       # normal
        meter_rows("ZN.4-PB_child__1_ch3pm", 1000 * weekly * (1 + 0.02 * rng.normal(size=DAYS))),      # same, x10
        meter_rows("ZN.5-PB_child__1_ch3pm", 100 * weekly * (1 + 0.02 * rng.normal(size=DAYS)),
                   untrusted=(t >= DAYS - 20)),                                                        # outage in the last 20 days
    ]
    daily = pd.concat(meters, ignore_index=True)
    short = daily[~((daily["meter_label"] == "ZN.1-PB_child__1_ch3pm") & (daily["date"] < START + pd.Timedelta(days=DAYS - 40)))]
    panel = Panel(short, synthetic_weather(), START + pd.Timedelta(days=DAYS + 90))
    issue = panel.index_of(START + pd.Timedelta(days=DAYS))
    model = Forecaster().fit(panel, issue)
    rows = origin_rows(panel, issue, horizons=60)
    rows["pred"] = model.predict_kwh(rows)
    rows["label"] = np.array(panel.meters)[rows["meter_idx"]]
    return rows


def level(rows, label, lo, hi):
    part = rows[(rows["label"] == label) & rows["days_ahead"].between(lo, hi)]
    return part["pred"].mean()


def test_every_forecast_is_finite_and_never_negative(synthetic):
    assert np.isfinite(synthetic["pred"]).all() and (synthetic["pred"] >= 0).all()


def test_a_constant_meter_is_forecast_as_constant(synthetic):
    for lo, hi in ((0, 6), (7, 29), (30, 59)):
        assert level(synthetic, "ZN.1-PB_child__1_ch3pm", lo, hi) == pytest.approx(100, rel=0.08)


def test_a_meter_with_only_forty_days_of_history_still_gets_a_sane_forecast(synthetic):
    assert 70 < level(synthetic, "ZN.1-PB_child__1_ch3pm", 0, 59) < 130


def test_a_dead_meter_stays_at_zero_and_does_not_invent_energy(synthetic):
    assert level(synthetic, "ZN.2-PB_child__1_ch3pm", 0, 59) < 1.0


def test_scaling_a_meters_history_scales_its_forecast(synthetic):
    ratio = level(synthetic, "ZN.4-PB_child__1_ch3pm", 0, 59) / level(synthetic, "ZN.3-PB_child__1_ch3pm", 0, 59)
    assert ratio == pytest.approx(10, rel=0.15)


def test_a_trailing_outage_falls_back_to_the_last_trusted_level(synthetic):
    assert level(synthetic, "ZN.5-PB_child__1_ch3pm", 0, 29) == pytest.approx(100 * 1.045, rel=0.2)
    base_age = synthetic[synthetic["label"] == "ZN.5-PB_child__1_ch3pm"]["base_age_days"].iloc[0]
    assert base_age >= 19


def test_features_are_finite_for_every_row_the_model_scores(synthetic):
    numeric = synthetic.select_dtypes("number").drop(columns=["actual_kwh", "target"], errors="ignore")
    assert not np.isinf(numeric.to_numpy(float)).any()


# ---- Part 2: perturb the real data and refit ---------------------------------------------

def forecast(daily, weather, forecast_end, tmp_path):
    service = ForecastService(ArtifactRepository(tmp_path))
    panel = service.build_panel(daily, weather, forecast_end)
    future, _, issue = service.forecast_to_year_end(panel, daily, forecast_end)
    return future, issue


@pytest.fixture(scope="module")
def real_baseline(inputs_module, tmp_path_factory):
    daily, weather, end = inputs_module
    return forecast(daily, weather, end, tmp_path_factory.mktemp("base"))[0]


@pytest.fixture(scope="module")
def inputs_module():
    from tests.conftest import DAILY, WEATHER
    if not (DAILY.exists() and WEATHER.exists()):
        pytest.skip("run run_pipeline.py first")
    daily, weather = pd.read_parquet(DAILY), pd.read_csv(WEATHER, parse_dates=["date"])
    last = daily.loc[daily["complete_day"], "date"].max()
    return daily, weather, pd.Timestamp(year=last.year, month=12, day=31)


def last_days(daily, n):
    return sorted(daily.loc[daily["complete_day"], "date"].unique())[-n:]


def fully_trusted_meters(daily, n=7):
    """Meters whose last n complete days are all trusted, so corrupting them really reaches the model."""
    tail = daily[daily["date"].isin(last_days(daily, n))].pivot_table(
        index="meter_label", columns="date", values="measurable_day", aggfunc="first")
    return tail.index[tail.sum(axis=1) == n].tolist()


@pytest.mark.slow
@needs_data
def test_changing_future_data_after_the_cutoff_does_not_change_a_trained_model(inputs_module):
    """Label-leak attack: train twice, the second time with every value after the cutoff scrambled."""
    from models.forecasting.features import Panel as P
    daily, weather, end = inputs_module
    cutoff_day = pd.Timestamp("2026-06-01")
    a = P(daily, weather, end)
    rng = np.random.default_rng(7)
    later = daily["date"] >= cutoff_day
    scrambled = daily.copy()
    for col in ("gold_kwh", "spread_kwh"):
        scrambled.loc[later, col] = rng.uniform(0, 8000, later.sum())
    scrambled.loc[later, "measurable_day"] = True
    hot = weather.copy()
    hot.loc[hot["date"] >= cutoff_day, "t_mean"] = 70.0
    b = P(scrambled, hot, end)
    cut = a.index_of(cutoff_day)
    ca, cb = Forecaster().fit(a, cut).coefficients(), Forecaster().fit(b, cut).coefficients()
    pd.testing.assert_frame_equal(ca, cb)


@pytest.mark.slow
@needs_data
def test_an_outage_in_the_last_week_moves_the_forecast_but_not_beyond_a_factor_of_two(inputs_module, real_baseline, tmp_path):
    """Known sensitivity: losing the latest week sends the model back to older data (measured: 0.5x to 1.2x)."""
    daily, weather, end = inputs_module
    hit = fully_trusted_meters(daily)[:3]
    assert len(hit) == 3
    broken = daily.copy()
    broken.loc[broken["meter_label"].isin(hit) & broken["date"].isin(last_days(daily, 7)), "measurable_day"] = False
    future, _ = forecast(broken, weather, end, tmp_path)
    ratio = (future.groupby("meter_label")["forecast_kwh"].sum()
             / real_baseline.groupby("meter_label")["forecast_kwh"].sum())[hit]
    assert np.isfinite(future["forecast_kwh"]).all()
    assert (ratio - 1).abs().max() > 0.01, "the corruption never reached the model (vacuous test)"
    assert ratio.between(0.4, 2.5).all(), ratio


@pytest.mark.slow
@needs_data
def test_a_tenfold_spike_in_the_last_days_is_followed_but_bounded(inputs_module, real_baseline, tmp_path):
    """Known weakness: a 3-day sensor spike lifts the whole horizon (measured: 8x tomorrow, 3.8x at the end)."""
    daily, weather, end = inputs_module
    label = fully_trusted_meters(daily)[0]
    spiky = daily.copy()
    mask = (spiky["meter_label"] == label) & spiky["date"].isin(last_days(daily, 3))
    for col in ("gold_kwh", "spread_kwh"):
        spiky.loc[mask, col] *= 10
    future, _ = forecast(spiky, weather, end, tmp_path)
    got = future[future["meter_label"] == label].sort_values("date")["forecast_kwh"].to_numpy()
    base = real_baseline[real_baseline["meter_label"] == label].sort_values("date")["forecast_kwh"].to_numpy()
    assert np.isfinite(got).all()
    assert got[0] / base[0] > 3, "the spike never reached the model (vacuous test)"
    assert (got / base).max() <= 11 and got.sum() / base.sum() < 6          # never beyond the 10x that was injected


@pytest.mark.slow
@needs_data
def test_a_twenty_degree_heatwave_in_the_last_week_is_bounded(inputs_module, real_baseline, tmp_path):
    daily, weather, end = inputs_module
    hot = weather.copy()
    days = last_days(daily, 7)
    hot.loc[hot["date"].isin(days), ["t_mean", "t_max", "t_min"]] += 20
    future, _ = forecast(daily, hot, end, tmp_path)
    ratio = future["forecast_kwh"].sum() / real_baseline["forecast_kwh"].sum()
    assert np.isfinite(ratio) and 0.5 < ratio < 2.5
