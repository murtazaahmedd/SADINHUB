"""Riyadh daily temperature: observed history and a day-of-year climate normal.

Source: Open-Meteo historical weather API (ERA5 reanalysis), free and keyless.
Only the site coordinates are sent. Results are cached to
outputs/data/weather_riyadh_daily.csv so the pipeline runs offline afterwards.

How the model may use it (no future information leaks into a forecast):
  * observed temperature up to the forecast issue date (yesterday and before);
  * the climate normal for any date, built from years *before* the forecast year,
    which is what anyone could know in advance about a future day.
"""
import json
import urllib.request
from datetime import date, timedelta

import numpy as np
import pandas as pd

from . import config

CACHE = config.DATA_DIR / "weather_riyadh_daily.csv"
HISTORY_START = date(2010, 1, 1)
_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
_FORECAST = "https://api.open-meteo.com/v1/forecast"
_FIELDS = "temperature_2m_max,temperature_2m_min,temperature_2m_mean"


def _fetch(url, start, end):
    query = (f"{url}?latitude={config.SITE_LATITUDE}&longitude={config.SITE_LONGITUDE}"
             f"&start_date={start}&end_date={end}&daily={_FIELDS}&timezone=Asia%2FRiyadh")
    with urllib.request.urlopen(query, timeout=60) as response:
        payload = json.loads(response.read().decode())
    daily = payload["daily"]
    return pd.DataFrame({"date": pd.to_datetime(daily["time"]),
                         "t_max": daily["temperature_2m_max"],
                         "t_min": daily["temperature_2m_min"],
                         "t_mean": daily["temperature_2m_mean"]})


def load_weather(until, refresh=False):
    """Observed daily temperature from 2010 through ``until`` (inclusive)."""
    until = pd.Timestamp(until).date()
    if CACHE.exists() and not refresh:
        cached = pd.read_csv(CACHE, parse_dates=["date"])
        if cached["date"].max().date() >= until:
            return cached
    frames = []
    year_start = HISTORY_START
    while year_start <= until:
        year_end = min(date(year_start.year, 12, 31), until)
        frames.append(_fetch(_ARCHIVE, year_start, year_end))
        year_start = date(year_start.year + 1, 1, 1)
    out = pd.concat(frames, ignore_index=True).dropna(subset=["t_max"])
    # The reanalysis archive lags real time by a few days; fill the tail from
    # the forecast service's recent past (observations, not forecasts).
    last = out["date"].max().date()
    if last < until:
        recent = _fetch(_FORECAST, last + timedelta(days=1), until).dropna(subset=["t_max"])
        out = pd.concat([out, recent], ignore_index=True)
    out = out.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    out.to_csv(CACHE, index=False)
    return out


def climate_normal(weather, before_year):
    """Smoothed day-of-year normal of t_mean/t_max using only years < before_year."""
    past = weather[weather["date"].dt.year < before_year].copy()
    past["doy"] = past["date"].dt.dayofyear.clip(upper=365)
    normal = past.groupby("doy")[["t_mean", "t_max"]].mean().reindex(range(1, 366))
    # Circular smoothing over +/-7 days removes single-year noise.
    padded = pd.concat([normal.iloc[-7:], normal, normal.iloc[:7]])
    smooth = padded.rolling(15, center=True, min_periods=1).mean().iloc[7:-7]
    smooth.index.name = "doy"
    return smooth.rename(columns={"t_mean": "normal_t_mean", "t_max": "normal_t_max"})


def cooling_degree_days(t_mean, base=18.0):
    return np.maximum(np.asarray(t_mean, dtype=float) - base, 0.0)
