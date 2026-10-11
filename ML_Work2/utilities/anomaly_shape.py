"""Night base load: the power a meter draws between midnight and 05:00, when little should be running."""
import pandas as pd

from constants.anomaly import NIGHT_END_HOUR, NIGHT_MIN_HOURS
from utilities.guard import require, traced

_BAD = ["UNRESOLVED_DECREMENT", "REGISTER_JUMP"]


@traced
def night_base_load(intervals: pd.DataFrame) -> pd.DataFrame:
    """
    #Feature: Night base load
    #Scenario: Average kW per meter and day between midnight and 05:00 Saudi time
    #Complexity: Time O(n) - one pass over the five-minute intervals

    Only normal intervals count (long gaps and impossible register moves are excluded), and a night needs at
    least three hours of readings to be trusted.

    Args:
        intervals: Validated intervals with meter_label, saudi, date, consumption_kwh, gap_minutes,
            interval_quality and status.

    Returns:
        Columns meter_label, date, night_hours and night_kw.

    Raises:
        ValueError: When intervals is None.
    """
    require("night_base_load", "intervals", intervals)
    cols = ["meter_label", "saudi", "date", "consumption_kwh", "gap_minutes", "interval_quality", "status"]
    iv = intervals[cols]
    iv = iv[(iv["interval_quality"] == "NORMAL_INTERVAL") & ~iv["status"].isin(_BAD) & (iv["saudi"].dt.hour < NIGHT_END_HOUR)]
    g = iv.groupby(["meter_label", "date"]).agg(kwh=("consumption_kwh", "sum"), minutes=("gap_minutes", "sum")).reset_index()
    g["night_hours"] = g["minutes"] / 60.0
    g = g[g["night_hours"] >= NIGHT_MIN_HOURS].copy()
    g["night_kw"] = g["kwh"] / g["night_hours"]
    return g[["meter_label", "date", "night_hours", "night_kw"]]
