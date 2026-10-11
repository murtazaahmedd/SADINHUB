"""Forecast from the latest data to 31 December 2026."""
import numpy as np
import pandas as pd

from . import config
from .features import origin_rows
from .model import Forecaster


def long_range_check(panel, daily, first="2025-09-01"):
    """How accurate were facility forecasts 1-30, 31-60 and 61-90 days ahead?

    Replays a forecast on the 1st of every month (model trained only on data
    before that day) and compares each 30-day block with the actual facility
    kWh. Outage catch-ups are spread over the days they belong to, so a meter
    returning from a long outage does not distort the comparison.
    """
    actual = (daily[daily["complete_day"]].pivot_table(index="meter_label", columns="date", values="spread_kwh")
              .reindex(index=panel.meters, columns=panel.dates).to_numpy())
    last_actual = panel.index_of(daily.loc[daily["complete_day"], "date"].max())
    rows = []
    for start in pd.date_range(first, panel.dates[last_actual], freq="MS"):
        t = panel.index_of(start)
        if t + 30 > last_actual + 1:
            break
        model = Forecaster().fit(panel, t)
        pred = origin_rows(panel, t, horizons=90)
        pred["kwh"] = model.predict_kwh(pred)
        for lo, label in ((0, "1-30 days ahead"), (30, "31-60 days ahead"), (60, "61-90 days ahead")):
            if t + lo + 30 > last_actual + 1:
                continue
            forecast = pred.loc[pred["days_ahead"].between(lo, lo + 29), "kwh"].sum()
            real = np.nansum(actual[:, t + lo:t + lo + 30])
            rows.append({"issued": f"{start:%Y-%m-%d}", "window": label,
                         "forecast_kwh": round(forecast, 1), "actual_kwh": round(real, 1)})
    out = pd.DataFrame(rows)
    out["accuracy"] = (100 * (1 - (out["forecast_kwh"] - out["actual_kwh"]).abs() / out["actual_kwh"])).round(1)
    overall = out.groupby("window").apply(
        lambda g: round(100 * (1 - (g["forecast_kwh"] - g["actual_kwh"]).abs().sum() / g["actual_kwh"].sum()), 1),
        include_groups=False)
    return out, overall.to_dict()


def forecast_to_year_end(panel, daily):
    """Train on every complete day, then forecast each day to FORECAST_END.

    The forecast is issued on the morning after the last complete day, exactly
    like each morning of the validation.
    """
    last_complete = daily.loc[daily["complete_day"], "date"].max()
    issue = last_complete + pd.Timedelta(days=1)
    t = panel.index_of(issue)
    horizons = panel.index_of(config.FORECAST_END) - t + 1
    model = Forecaster().fit(panel, t)
    rows = origin_rows(panel, t, horizons=horizons)
    rows["forecast_kwh"] = model.predict_kwh(rows)
    out = pd.DataFrame({
        "date": panel.dates[rows["target_idx"].to_numpy()],
        "meter_label": np.array(panel.meters)[rows["meter_idx"].to_numpy()],
        "zone": panel.zone[rows["meter_idx"].to_numpy()],
        "days_ahead": rows["days_ahead"].to_numpy() + 1,
        "forecast_kwh": rows["forecast_kwh"].round(2),
    })
    return out, model, issue
