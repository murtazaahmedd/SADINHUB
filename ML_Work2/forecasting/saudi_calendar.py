"""Saudi working calendar: weekend, Ramadan, Eid holidays and national days.

Islamic dates follow the Umm al-Qura calendar (moon-sighting can move them by
a day; edit the table if the official announcement differs). These are calendar
inputs that are known in advance, exactly like a public-holiday list - they are
not tuned to any result.
"""
import pandas as pd

# (first day, last day) inclusive.
RAMADAN = [("2025-03-01", "2025-03-29"), ("2026-02-18", "2026-03-19"), ("2027-02-08", "2027-03-09")]
EID_AL_FITR_HOLIDAY = [("2025-03-30", "2025-04-02"), ("2026-03-20", "2026-03-23"), ("2027-03-10", "2027-03-13")]
EID_AL_ADHA_HOLIDAY = [("2025-06-05", "2025-06-09"), ("2026-05-26", "2026-05-30")]
FIXED_HOLIDAYS = {(2, 22): "founding_day", (9, 23): "national_day"}

# Saudi weekend is Friday and Saturday (pandas: Monday=0 ... Friday=4, Saturday=5).
WEEKEND_DAYS = (4, 5)


def _in_ranges(dates, ranges):
    mask = pd.Series(False, index=dates.index)
    for start, end in ranges:
        mask |= dates.between(pd.Timestamp(start), pd.Timestamp(end))
    return mask


def _days_to_range_start(dates, ranges):
    """Signed days from each date to the nearest range start (negative = before)."""
    starts = pd.to_datetime([s for s, _ in ranges])
    diffs = [(dates - s).dt.days for s in starts]
    frame = pd.concat(diffs, axis=1)
    pick = frame.abs().idxmin(axis=1)
    return frame.to_numpy()[range(len(frame)), pick.to_numpy()]


def calendar_features(dates):
    """One row of calendar facts per date (index aligned with ``dates``)."""
    dates = pd.Series(pd.to_datetime(dates))
    out = pd.DataFrame(index=dates.index)
    out["dow"] = dates.dt.dayofweek
    out["weekend"] = dates.dt.dayofweek.isin(WEEKEND_DAYS).astype(int)
    out["ramadan"] = _in_ranges(dates, RAMADAN).astype(int)
    out["eid"] = (_in_ranges(dates, EID_AL_FITR_HOLIDAY) | _in_ranges(dates, EID_AL_ADHA_HOLIDAY)).astype(int)
    fixed = [(m, d) for m, d in zip(dates.dt.month, dates.dt.day)]
    out["national_holiday"] = [int(md in FIXED_HOLIDAYS) for md in fixed]
    out["off_day"] = ((out["weekend"] + out["eid"] + out["national_holiday"]) > 0).astype(int)
    out["days_from_ramadan_start"] = _days_to_range_start(dates, RAMADAN)
    out["days_from_eid_adha"] = _days_to_range_start(dates, EID_AL_ADHA_HOLIDAY)
    out["day_of_year"] = dates.dt.dayofyear
    return out
