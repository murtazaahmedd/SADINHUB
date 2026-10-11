"""End-to-end run: data -> validation of 2026 -> forecast to year end -> files."""
import json
import time

import numpy as np
import pandas as pd

from . import config
from .actuals import build_daily_actuals
from .backtest import walk_forward
from .billing import load_breakers
from .features import Panel
from .future import forecast_to_year_end, long_range_check
from .scoring import all_levels, calendar_months, meter_periods, month_matrix, summary
from .weather import load_weather

HORIZON_LABELS = {1: "1 day", 7: "7 days", 30: "30 days", "month": "calendar month"}
# Every forecast made on every morning of 2026 (meter, morning, days ahead, kWh).
WALK_FORWARD_PATH = config.VALIDATION_DIR / "every_morning_forecast_2026.parquet"


def _say(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _score(panel, preds):
    periods = pd.concat([meter_periods(panel, preds), calendar_months(panel, preds)], ignore_index=True)
    periods["horizon"] = periods["horizon"].map(HORIZON_LABELS)
    return periods, all_levels(periods)


def saved_results():
    """Reload the last full run (no retraining) - used to rebuild the dashboard."""
    daily = build_daily_actuals()
    weather = load_weather(daily["date"].max())
    panel = Panel(daily, weather, config.FORECAST_END)
    preds = pd.read_parquet(WALK_FORWARD_PATH)
    _, levels = _score(panel, preds)
    future = pd.read_csv(config.FORECAST_DIR / "forecast_daily_meters_to_2026-12-31.csv", parse_dates=["date"])
    info = json.loads((config.FORECAST_DIR / "forecast_info.json").read_text())
    issue = pd.Timestamp(info["forecast_issued_for_morning_of"])
    long_detail = pd.read_csv(config.VALIDATION_DIR / "long_range_accuracy_facility.csv")
    long_overall = long_detail.groupby("window").apply(
        lambda g: round(100 * (1 - (g["forecast_kwh"] - g["actual_kwh"]).abs().sum() / g["actual_kwh"].sum()), 1),
        include_groups=False).to_dict()
    return panel, daily, preds, levels, future, issue, long_overall


def run(refresh_sources=False):
    _say("1/5 Reading meter data (database + Excel) and building daily kWh")
    daily = build_daily_actuals(refresh=True, refresh_sources=refresh_sources)
    last_complete = daily.loc[daily["complete_day"], "date"].max()
    _say(f"    {daily['meter_label'].nunique()} meters, data to {last_complete:%Y-%m-%d}")

    _say("2/5 Riyadh temperature history and seasonal normals")
    weather = load_weather(daily["date"].max())
    panel = Panel(daily, weather, config.FORECAST_END)

    _say("3/5 Replaying every morning of 2026 (walk-forward validation)")
    preds, coefficients = walk_forward(panel, config.VALIDATION_START, last_complete, horizons=31, log=_say)

    preds.to_parquet(WALK_FORWARD_PATH, index=False)

    _say("4/5 Scoring against gold actuals")
    periods, levels = _score(panel, preds)
    _write_validation(panel, preds, periods, levels, coefficients)

    _say("5/5 Forecasting to 31 December 2026 (and checking how far ahead it stays accurate)")
    future, model, issue = forecast_to_year_end(panel, daily)
    _write_future(daily, future, model, issue)
    long_detail, long_overall = long_range_check(panel, daily)
    long_detail.to_csv(config.VALIDATION_DIR / "long_range_accuracy_facility.csv", index=False)
    _say(f"    facility accuracy by distance ahead: {long_overall}")
    load_breakers(refresh=True)   # breaker ratings for the bill forecast, by meter key
    _say("Done.")
    return panel, daily, preds, levels, future, issue, long_overall


def _write_validation(panel, preds, periods, levels, coefficients):
    out = config.VALIDATION_DIR
    detail = levels.copy()
    days = detail["horizon"].map({"1 day": 1, "7 days": 7, "30 days": 30})
    month_len = detail["issue_date"].dt.days_in_month
    detail["period_start"] = detail["issue_date"]
    detail["period_end"] = detail["issue_date"] + pd.to_timedelta(days.fillna(month_len) - 1, unit="D")
    detail = detail[["level", "unit", "horizon", "issue_date", "period_start", "period_end",
                     "forecast", "actual", "abs_error", "accuracy", "meters"]]
    detail.round({"forecast": 2, "actual": 2, "abs_error": 2, "accuracy": 2}).to_csv(
        out / "validation_detail_2026.csv", index=False)

    summ = summary(levels)
    summ.to_csv(out / "accuracy_summary_2026.csv", index=False)
    for level in ("meter", "zone", "facility"):
        for horizon in HORIZON_LABELS.values():
            matrix = month_matrix(levels, level, horizon).round(1)
            matrix.to_csv(out / f"accuracy_by_month_{level}_{horizon.replace(' ', '_')}.csv")

    excluded = periods[~periods["measurable"]]
    excluded[["meter_label", "horizon", "issue_date", "forecast", "actual"]].round({"forecast": 2, "actual": 2}).to_csv(
        out / "periods_not_measurable_data_gaps.csv", index=False)
    coef = pd.concat({month: frame for month, frame in coefficients.items()}, names=["model_month", "driver"])
    coef.round(4).to_csv(out / "learned_model_weights_by_month.csv")

    # Day-ahead forecasts next to gold daily actuals, for charts and audit.
    day0 = preds[preds["days_ahead"] == 0]
    daily_view = pd.DataFrame({
        "date": panel.dates[day0["target_idx"].to_numpy()],
        "meter_label": np.array(panel.meters)[day0["meter_idx"].to_numpy()],
        "zone": panel.zone[day0["meter_idx"].to_numpy()],
        "day_ahead_forecast_kwh": day0["pred_kwh"].round(2).to_numpy(),
        "actual_kwh": panel.gold[day0["meter_idx"].to_numpy(), day0["target_idx"].to_numpy()].round(2),
        "measurable_day": panel.measurable[day0["meter_idx"].to_numpy(), day0["target_idx"].to_numpy()],
    })
    daily_view.to_csv(out / "day_ahead_forecast_vs_actual_2026.csv", index=False)
    (out / "accuracy_summary_2026.json").write_text(json.dumps(summ.to_dict("records"), indent=2, default=str))


def _write_future(daily, future, model, issue):
    out = config.FORECAST_DIR
    future.to_csv(out / "forecast_daily_meters_to_2026-12-31.csv", index=False)
    zone = future.groupby(["date", "zone"], as_index=False)["forecast_kwh"].sum()
    zone.to_csv(out / "forecast_daily_zones_to_2026-12-31.csv", index=False)
    fac = future.groupby("date", as_index=False)["forecast_kwh"].sum()
    fac.to_csv(out / "forecast_daily_facility_to_2026-12-31.csv", index=False)

    actual = daily[daily["complete_day"] & (daily["date"].dt.year == 2026)]
    act_m = actual.groupby([actual["date"].dt.strftime("%Y-%m"), "meter_label"])["gold_kwh"].sum()
    fut_m = future.groupby([future["date"].dt.strftime("%Y-%m"), "meter_label"])["forecast_kwh"].sum()
    monthly = pd.concat({"actual_kwh": act_m, "forecast_kwh": fut_m}, axis=1).fillna(0.0)
    monthly["outlook_kwh"] = monthly["actual_kwh"] + monthly["forecast_kwh"]
    monthly.index.names = ["month", "meter_label"]
    monthly.round(1).to_csv(out / "monthly_outlook_2026_actual_plus_forecast.csv")
    model.coefficients().round(4).to_csv(out / "model_weights_used_for_forecast.csv")
    (out / "forecast_info.json").write_text(json.dumps({
        "forecast_issued_for_morning_of": f"{issue:%Y-%m-%d}",
        "uses_actuals_through": f"{issue - pd.Timedelta(days=1):%Y-%m-%d}",
        "forecast_until": f"{config.FORECAST_END:%Y-%m-%d}",
        "model_trained_through": f"{model.trained_through:%Y-%m-%d}",
        "training_examples": model.training_rows,
    }, indent=2))
