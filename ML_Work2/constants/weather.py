"""Open-Meteo weather source literals."""
from datetime import date

HISTORY_START: date = date(2010, 1, 1)
ARCHIVE_URL: str = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL: str = "https://api.open-meteo.com/v1/forecast"
DAILY_FIELDS: str = "temperature_2m_max,temperature_2m_min,temperature_2m_mean"
REQUEST_TIMEOUT_SECONDS: int = 60
NORMAL_SMOOTHING_DAYS: int = 15
