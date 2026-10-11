"""Project-wide settings: folders, time basis and business constants.

Everything that a reader might want to change lives here, so no number used by
the model is buried inside the code.
"""
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INPUT_EXCEL = ROOT / "input_excel"
OUTPUTS = ROOT / "outputs"
DATA_DIR = OUTPUTS / "data"
VALIDATION_DIR = OUTPUTS / "validation"
FORECAST_DIR = OUTPUTS / "forecasts"
MODEL_DIR = OUTPUTS / "models"
DASHBOARD_DIR = ROOT / "dashboard"

for _folder in (DATA_DIR, VALIDATION_DIR, FORECAST_DIR, MODEL_DIR, DASHBOARD_DIR):
    _folder.mkdir(parents=True, exist_ok=True)

# Saudi Arabia is fixed UTC+3 with no daylight saving. Every date in this
# project is a Saudi calendar date, the same basis as gold.reading_date_saudi.
SAUDI_OFFSET_HOURS = 3

# Gold-layer rule: an interval of 15 minutes or more is a LONG_GAP.
LONG_GAP_MINUTES = 15.0

# Gold-layer rule: two meters have installation readings that are excluded.
# Timestamps are UTC, exactly as stored in silver.
STARTUP_EXCLUDED_BEFORE_UTC = {
    "ZN.1-PB.1_child__1_ch3pm": datetime(2024, 12, 10, 17, 16, 15, 655000),
    "ZN.4-PB.1_child__1_ch3pm": datetime(2024, 12, 10, 21, 46, 39, 10000),
}

# Physical plausibility: the whole facility's coincident peak is ~482 kW, so a
# single meter whose register rises faster than this inside a normal interval
# has jumped (re-commissioning / register swap), it did not consume that energy.
# Only Zone 13 ever triggers it (one reading on 2025-02-05); gold is unaffected.
MAX_PLAUSIBLE_METER_KW = 500.0

# A day is "measurable" for day-level accuracy only when the readings cover
# almost all of it, so the booked kWh is the energy that was really used that day.
MIN_DAY_COVERAGE_SHARE = 0.95
# A gap longer than this that crosses a midnight moves energy between days,
# so period totals whose boundary falls inside such a gap are not measurable.
BOUNDARY_GAP_MINUTES = 60.0

# Riyadh site coordinates, from gold.zone.
SITE_LATITUDE = 24.69750933
SITE_LONGITUDE = 46.66536212

# Validation and forecast calendar.
VALIDATION_START = date(2026, 1, 1)
FORECAST_END = date(2026, 12, 31)
HORIZONS = (1, 7, 30)
ACCURACY_TARGET = 80.0

# Database: read-only, password from the environment, never stored in a file.
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "sadinhub"
DB_USER = "postgres"
