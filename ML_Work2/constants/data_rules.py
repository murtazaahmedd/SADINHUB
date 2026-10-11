"""Business rules and literals for turning raw meter readings into daily kWh."""
import re
from datetime import datetime
from typing import Dict

# Saudi Arabia is fixed UTC+3 with no daylight saving. Every date in this
# project is a Saudi calendar date, the same basis as gold.reading_date_saudi.
SAUDI_OFFSET_HOURS: int = 3

# Gold-layer rule: an interval of 15 minutes or more is a LONG_GAP.
LONG_GAP_MINUTES: float = 15.0

# Gold-layer rule: meters whose installation readings are excluded.
# Timestamps are UTC, exactly as stored in silver.
STARTUP_EXCLUDED_BEFORE_UTC: Dict[str, datetime] = {
    "ZN.1-PB.1_child__1_ch3pm": datetime(2024, 12, 10, 17, 16, 15, 655000),
    "ZN.4-PB.1_child__1_ch3pm": datetime(2024, 12, 10, 21, 46, 39, 10000),
}

# Physical plausibility: a single meter whose register rises faster than this
# inside a normal interval has jumped (re-commissioning / register swap); it
# did not consume that energy. The facility's coincident peak is ~482 kW.
MAX_PLAUSIBLE_METER_KW: float = 500.0

# A day is "measurable" only when the readings cover almost all of it.
MIN_DAY_COVERAGE_SHARE: float = 0.95
# A gap longer than this that crosses a midnight moves energy between days.
BOUNDARY_GAP_MINUTES: float = 60.0

# Telemetry export file name: ..._ZN.<zone>-PB[.<n>]_child_<n>_ch3pm_...
EXCEL_FILE_LABEL = re.compile(r"_(ZN\.\d+-PB(?:\.\d+)?)_child_(\d+)_ch3pm_")
# An export continues a meter's register if the kWh step is within this rate.
EXCEL_MAX_CONTINUITY_KWH_PER_HOUR: float = 60.0

# Telemetry key that holds the cumulative active energy register.
ENERGY_TELEMETRY_KEY: str = "kWh"
