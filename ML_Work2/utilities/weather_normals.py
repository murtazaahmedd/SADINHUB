"""Day-of-year climate normals built only from years before the forecast year."""
import pandas as pd

from constants.weather import NORMAL_SMOOTHING_DAYS
from utilities.guard import require, traced


@traced
def climate_normal(weather: pd.DataFrame, before_year: int) -> pd.DataFrame:
    """
    #Feature: Weather normals
    #Scenario: What anyone could know in advance about a future day's temperature
    #Complexity: Time O(n) - linear in the weather history

    Smoothed day-of-year normal of t_mean and t_max using only years < before_year.
    Circular smoothing over +/-7 days removes single-year noise.

    Args:
        weather: Daily observed temperature with date, t_mean and t_max.
        before_year: Only years strictly before this are used.

    Returns:
        DataFrame indexed by day of year (1-365) with normal_t_mean and normal_t_max.

    Raises:
        ValueError: When weather or before_year is None.
    """
    require("climate_normal", "weather", weather)
    require("climate_normal", "before_year", before_year)
    past = weather[weather["date"].dt.year < before_year].copy()
    past["doy"] = past["date"].dt.dayofyear.clip(upper=365)
    normal = past.groupby("doy")[["t_mean", "t_max"]].mean().reindex(range(1, 366))
    pad = NORMAL_SMOOTHING_DAYS // 2
    padded = pd.concat([normal.iloc[-pad:], normal, normal.iloc[:pad]])
    smooth = padded.rolling(NORMAL_SMOOTHING_DAYS, center=True, min_periods=1).mean().iloc[pad:-pad]
    smooth.index.name = "doy"
    return smooth.rename(columns={"t_mean": "normal_t_mean", "t_max": "normal_t_max"})
