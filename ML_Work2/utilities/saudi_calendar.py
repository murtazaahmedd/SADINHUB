"""Saudi working calendar: weekend, Ramadan, Eid holidays and national days.

These are calendar inputs known in advance, exactly like a public-holiday list;
they are not tuned to any result.
"""
import datetime as dt
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from hijridate import Gregorian, Hijri

from constants.calendar import (DHUL_HIJJAH_MONTH, EID_AL_ADHA_FIRST_DAY, EID_AL_ADHA_HOLIDAY_DAYS,
                                EID_AL_FITR_HOLIDAY_DAYS, FIXED_HOLIDAYS, RAMADAN_MONTH, SHAWWAL_MONTH, WEEKEND_DAYS)
from utilities.guard import require, traced

Ranges = Sequence[Tuple[str, str]]


@lru_cache(maxsize=16)
def islamic_ranges(first_year: int, last_year: int) -> Dict[str, Tuple[Tuple[str, str], ...]]:
    """
    #Feature: Saudi calendar
    #Scenario: Ramadan and the two Eid holidays for every year in a range, from the Umm al-Qura calendar
    #Complexity: Time O(y) - years in the range (cached)

    Ramadan runs from 1 Ramadan to the last day of the month; Eid al-Fitr is 1-4 Shawwal;
    the Eid al-Adha holiday is 9-13 Dhul Hijjah. Moon-sighting can move a date by a day.

    Args:
        first_year: First Gregorian year to cover.
        last_year: Last Gregorian year to cover.

    Returns:
        {"ramadan": ..., "fitr": ..., "adha": ...} -> (first day, last day) ISO pairs, in date order.

    Raises:
        ValueError: When a year is outside the calendar's range.
    """
    one = dt.timedelta(days=1)
    start_h = Gregorian(first_year, 1, 1).to_hijri().year
    end_h = Gregorian(last_year, 12, 31).to_hijri().year
    out: Dict[str, List[Tuple[str, str]]] = {"ramadan": [], "fitr": [], "adha": []}
    for hy in range(start_h, end_h + 1):
        ram = Hijri(hy, RAMADAN_MONTH, 1).to_gregorian()
        fitr = Hijri(hy, SHAWWAL_MONTH, 1).to_gregorian()
        adha = Hijri(hy, DHUL_HIJJAH_MONTH, EID_AL_ADHA_FIRST_DAY).to_gregorian()
        out["ramadan"].append((str(ram), str(fitr - one)))
        out["fitr"].append((str(fitr), str(fitr + dt.timedelta(days=EID_AL_FITR_HOLIDAY_DAYS - 1))))
        out["adha"].append((str(adha), str(adha + dt.timedelta(days=EID_AL_ADHA_HOLIDAY_DAYS - 1))))
    return {k: tuple(v) for k, v in out.items()}


@traced
def in_ranges(dates: pd.Series, ranges: Ranges) -> pd.Series:
    """
    #Feature: Saudi calendar
    #Scenario: Mark the dates that fall inside any inclusive date range
    #Complexity: Time O(n*r) - n dates, r ranges

    Args:
        dates: Datetime series.
        ranges: (first day, last day) pairs, both inclusive.

    Returns:
        Boolean series aligned with ``dates``.

    Raises:
        ValueError: When dates or ranges is None.
    """
    require("in_ranges", "dates", dates)
    require("in_ranges", "ranges", ranges)
    mask = pd.Series(False, index=dates.index)
    for start, end in ranges:
        mask |= dates.between(pd.Timestamp(start), pd.Timestamp(end))
    return mask


@traced
def days_to_range_start(dates: pd.Series, ranges: Ranges) -> np.ndarray:
    """
    #Feature: Saudi calendar
    #Scenario: Signed distance to the nearest Ramadan or Eid start
    #Complexity: Time O(n*r) - n dates, r ranges

    Args:
        dates: Datetime series.
        ranges: (first day, last day) pairs.

    Returns:
        Days from each date to the nearest range start (negative = before the start).

    Raises:
        ValueError: When dates or ranges is None.
    """
    require("days_to_range_start", "dates", dates)
    require("days_to_range_start", "ranges", ranges)
    starts = pd.to_datetime([s for s, _ in ranges])
    frame = pd.concat([(dates - s).dt.days for s in starts], axis=1)
    pick = frame.abs().idxmin(axis=1)
    return frame.to_numpy()[range(len(frame)), pick.to_numpy()]


@traced
def _fixed_holiday_flags(dates: pd.Series) -> List[int]:
    """
    #Feature: Saudi calendar
    #Scenario: Founding Day and National Day flags
    #Complexity: Time O(n) - linear

    Args:
        dates: Datetime series.

    Returns:
        1 for each date that is a fixed national holiday, else 0.

    Raises:
        ValueError: When dates is None.
    """
    require("_fixed_holiday_flags", "dates", dates)
    return [int((m, d) in FIXED_HOLIDAYS) for m, d in zip(dates.dt.month, dates.dt.day)]


@traced
def calendar_features(dates: Sequence) -> pd.DataFrame:
    """
    #Feature: Saudi calendar
    #Scenario: One row of calendar facts per date, aligned with the input
    #Complexity: Time O(n*r) - n dates, r holiday ranges

    Args:
        dates: Anything convertible to a datetime series.

    Returns:
        DataFrame with dow, weekend, ramadan, eid, national_holiday, off_day,
        days_from_ramadan_start, days_from_eid_adha and day_of_year.

    Raises:
        ValueError: When dates is None.
    """
    require("calendar_features", "dates", dates)
    dates = pd.Series(pd.to_datetime(dates))
    islamic = islamic_ranges(int(dates.dt.year.min()) - 1, int(dates.dt.year.max()) + 1)
    out = pd.DataFrame(index=dates.index)
    out["dow"] = dates.dt.dayofweek
    out["weekend"] = dates.dt.dayofweek.isin(WEEKEND_DAYS).astype(int)
    out["ramadan"] = in_ranges(dates, islamic["ramadan"]).astype(int)
    out["eid"] = (in_ranges(dates, islamic["fitr"]) | in_ranges(dates, islamic["adha"])).astype(int)
    out["national_holiday"] = _fixed_holiday_flags(dates)
    out["off_day"] = ((out["weekend"] + out["eid"] + out["national_holiday"]) > 0).astype(int)
    out["days_from_ramadan_start"] = days_to_range_start(dates, islamic["ramadan"])
    out["days_from_eid_adha"] = days_to_range_start(dates, islamic["adha"])
    out["day_of_year"] = dates.dt.dayofyear
    return out
