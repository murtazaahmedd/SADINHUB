"""Model, feature and scoring literals for the forecasting pipeline."""
from typing import Dict, List, Tuple

HORIZONS: Tuple[int, ...] = (1, 7, 30)
HORIZON_LABELS: Dict[object, str] = {1: "1 day", 7: "7 days", 30: "30 days", "month": "calendar month"}
ACCURACY_TARGET: float = 80.0

# Days ahead the model is trained for. A forecast needing more days raises.
MAX_HORIZON_DAYS: int = 180
# Days ahead replayed on every historical morning (covers a 30-day month).
WALK_FORWARD_DAYS: int = 31
LONG_RANGE_DAYS: int = 90
LONG_RANGE_WARMUP_MONTHS: int = 9   # first replay is 9 months after the data begins
ALERT_HISTORY_WARMUP_MONTHS: int = 4   # same-day forecasts for the alerts start 4 months after the data begins
LONG_RANGE_WINDOWS: Tuple[Tuple[int, str], ...] = (
    (0, "1-30 days ahead"), (30, "31-60 days ahead"), (60, "61-90 days ahead"))

# Feature windows (days).
BASE_WINDOW_DAYS: int = 7
LONG_WINDOW_DAYS: int = 28
YEAR_LAG_DAYS: int = 364        # 52 weeks keeps weekdays aligned
MIN_BASE_DAYS_SHORT: int = 4    # trusted days needed in the 7-day window
MIN_BASE_DAYS_LONG: int = 10    # trusted days needed in the 28-day window
FALLBACK_BASE_DAYS: int = 14
MIN_RECENT3_DAYS: int = 2
MIN_DOW_DAYS: int = 2
MIN_LAST_YEAR_DAYS: int = 3
MIN_TRAIN_ORIGIN: int = 8
RECENT_DAYS: int = 3            # "last 3 days" window
LAST_YEAR_HALF_WINDOW: int = 3  # days either side of the same day last year

# (first days-ahead, end days-ahead exclusive, train on every n-th morning, use tree)
SEGMENTS: List[Tuple[int, int, int, bool]] = [
    (0, 1, 1, True), (1, 7, 1, True), (7, 30, 2, False), (30, MAX_HORIZON_DAYS, 3, False)]

TREE_PARAMS: Dict[str, object] = dict(
    objective="l1", learning_rate=0.03, n_estimators=300, num_leaves=15,
    min_child_samples=300, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    reg_lambda=5.0, n_jobs=-1, verbose=-1, random_state=7,
)

HUBER_ITERATIONS: int = 15
HUBER_RIDGE: float = 1e-3
HUBER_TUNING: float = 1.345

FEATURES: List[str] = [
    "days_ahead", "meter_code", "log_base", "base_age_days", "trend_7_vs_28", "yesterday_vs_base",
    "two_days_ago_vs_base", "last_3_days_vs_base", "volatility_28",
    "dow_ratio", "same_day_last_week_vs_base", "last_year_change", "last_year_available",
    "dow", "weekend", "ramadan", "eid", "national_holiday", "off_day",
    "days_from_ramadan_start", "days_from_eid_adha",
    "base_ramadan_share", "base_off_share",
    "normal_temp_target", "normal_temp_shift", "recent_temp_anomaly", "recent_temp",
    "temp_effect_change", "seasonal_effect_change", "weather_anomaly_effect", "temp_sensitivity",
]
CATEGORICAL: List[str] = ["meter_code"]

# Temperature response: hinge points (deg C) and shrinkage toward the pooled curve.
TEMPERATURE_HINGES: Tuple[float, ...] = (15.0, 20.0, 25.0, 30.0, 35.0)
SHRINK_DAYS: float = 90.0
SENSITIVITY_TEMP_C: float = 38.0
MIN_RESPONSE_FIT_DAYS: int = 60
MIN_METER_FIT_DAYS: int = 30
POOLED_RIDGE: float = 1e-3
SCALE_EPSILON: float = 1e-9
