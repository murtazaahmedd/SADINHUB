"""Thresholds and literals for the anomaly detectors."""
from typing import Dict, List, Tuple

RULE_Z: float = 3.0
MIN_DIFF_KWH: float = 10.0
MIN_DIFF_SHARE: float = 0.20
FOREST: Dict[str, object] = dict(n_estimators=400, max_samples=256, contamination="auto", random_state=7)

# The ML detector's alert level is a fixed share of the checked meter-days in the validation period:
# 3% is about 6 alerts a week across 31 meters (217 meter-days a week), what a reviewer can work through.
# It is a staffing decision, not a number derived from the rule detector or from the test period.
ML_TARGET_ALERT_RATE: float = 0.03

RESIDUAL_WINDOW: int = 28        # checked days used for the usual error
MIN_RESIDUALS: int = 14          # minimum history before a meter-day can be scored
LEVEL_DAYS: int = 28             # calendar days for the recent level
MIN_LEVEL_DAYS: int = 7
FLOOR_KWH: float = 5.0
MIN_USUAL_ERROR_KWH: float = 2.0
MIN_USUAL_ERROR_SHARE: float = 0.05
MAD_SCALE: float = 1.4826

FEATURES: List[str] = ["z", "rel", "jump", "similar", "level"]
CLIP: Dict[str, float] = {"z": 20.0, "rel": 10.0, "jump": 10.0, "similar": 10.0, "level": 10.0}

# Period plan, as month/day boundaries of the forecast year (development starts at the first replayed day).
# (name, first month, first day, last month, last day)
# The ML alert level is calibrated on a long, typical window (Apr-Aug, five months incl. a noisy spring and a
# calm summer) rather than on the calmest two months, which made later months look twice as noisy as the target.
PERIOD_PLAN: List[Tuple[str, int, int, int, int]] = [
    ("development", 1, 1, 3, 31),
    ("validation", 4, 1, 8, 31),
    ("final test", 9, 1, 9, 30),
    ("latest", 10, 1, 12, 31),
]
VALIDATION_PERIOD: str = "validation"
DEVELOPMENT_PERIOD: str = "development"
FINAL_TEST_PERIOD: str = "final test"

SITE_WIDE_SHARE: float = 0.5
QUALITY_PARTIAL_COVERAGE: float = 0.95
GAP_HOURS_MINUTES: float = 60.0   # a reading gap longer than this counts as missing time

QUALITY_REASONS: Dict[str, str] = {
    "register": "register problem",
    "none": "no readings",
    "catchup": "catch-up booked after an outage",
    "partial": "incomplete readings",
}

EVENT_LEAD_DAYS: int = 7
EVENT_WEEK_DAYS: int = 7
SYNTHETIC_FACTORS: Dict[str, float] = {"60% higher": 1.6, "60% lower": 0.4}
MIN_DAYS_FOR_WEEKLY_RATE: int = 7

CHECK_TEXT: Dict[str, str] = {
    "higher": "Check: equipment left running, longer operating hours, a new tenant or equipment, or a meter fault.",
    "lower": "Check: closure or holiday, vacancy, equipment switched off, or a meter fault.",
    "switch_on": "Check: the meter started using power again after a quiet spell - new tenant or equipment switched on?",
    "near_zero": "Check: consumption fell almost to zero - is the area closed or the supply off?",
    "reopened": ("Check: consumption came back to this meter's normal level after a quiet spell; the morning "
                 "forecast expected the quiet spell to continue - confirm the area reopened (no action needed if so)."),
}

# ---- Incidents: consecutive alert days of one meter in one direction become one incident.
INCIDENT_MAX_GAP_DAYS: int = 2        # at most one quiet day between two alert days of the same incident
INCIDENT_ACTIVE_DAYS: int = 2         # an incident is "active" if its last alert is within this many days of the data end

# ---- Night base load: power drawn between midnight and 05:00 Saudi time.
NIGHT_END_HOUR: int = 5
NIGHT_MIN_HOURS: float = 3.0          # hours of normal readings needed in the window to trust a night value
NIGHT_WINDOW_DAYS: int = 28
NIGHT_MIN_HISTORY: int = 14
NIGHT_LOG_OFFSET_KW: float = 0.5      # log(kW + offset) keeps near-zero night loads well behaved
NIGHT_SPREAD_FLOOR: float = 0.15      # smallest usual variation, in log units
NIGHT_Z: float = 5.0                  # how many usual variations above the meter's own night level
NIGHT_MIN_RISE_KW: float = 1.0        # and at least this many kW above it
