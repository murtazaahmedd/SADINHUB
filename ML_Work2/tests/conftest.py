"""Shared fixtures. Tests that need saved pipeline results skip when they do not exist yet."""
import sys
import warnings
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore", category=RuntimeWarning)

from config.settings import Settings  # noqa: E402

DATA = ROOT / "outputs" / "data"
DAILY = DATA / "daily_actuals.parquet"
WEATHER = DATA / "weather_riyadh_daily.csv"
needs_data = pytest.mark.skipif(not (DAILY.exists() and WEATHER.exists()), reason="run run_pipeline.py first")


@pytest.fixture(scope="session")
def inputs():
    """The saved daily table and weather, as the pipeline wrote them."""
    return pd.read_parquet(DAILY), pd.read_csv(WEATHER, parse_dates=["date"])


@pytest.fixture(scope="session")
def forecast_end(inputs):
    """31 December of the year the data ends in (the same rule the pipeline uses)."""
    daily, _ = inputs
    return pd.Timestamp(year=int(daily.loc[daily["complete_day"], "date"].max().year), month=12, day=31)


@pytest.fixture()
def tmp_settings(tmp_path):
    """Settings whose outputs live in a temporary folder."""
    return Settings(root=tmp_path)


def alert_inputs(outputs=None):
    """The daily table, the meter-day frame and the period plan, built exactly as AnomalyService builds them."""
    from models.anomaly.detectors import period_plan
    from repositories.artifacts.artifact_repository import ArtifactRepository
    from utilities.anomaly_inputs import build_meter_days
    store = ArtifactRepository(outputs or ROOT / "outputs")
    daily = store.load_frame("daily_actuals")
    key = "every_morning_forecast_extended" if store.exists("every_morning_forecast_extended") else "every_morning_forecast"
    frame = build_meter_days(daily, store.load_frame(key), store.load_frame("validated_intervals"))
    return daily, frame, period_plan(int(daily.loc[daily["complete_day"], "date"].max().year), frame["date"].min().date())
