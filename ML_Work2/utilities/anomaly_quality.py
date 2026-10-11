"""Data-quality alerts: days whose readings cannot be trusted as consumption.

These are kept apart from high/low consumption alerts. A missing or broken
reading is never read as 0 kWh of consumption.

Reason for a single day (first that applies):
  register problem  - the cumulative register went backwards or jumped impossibly
  no readings       - the meter sent nothing all day
  catch-up booked   - the meter came back after an outage and the whole outage's
                      energy was booked on this day
  partial           - fewer than 95% of the day's minutes fall in normal (under
                      15-minute) reading intervals

Missing time is measured only from reading gaps longer than an hour. Consecutive
problem days of one meter form one episode. Days on which at least half of all
meters had a problem are reported once as a site-wide outage; a meter whose own
outage was already running when the site-wide day began stays in its own episode.
"""
from typing import Dict, List, Set, Tuple

import pandas as pd

from constants.anomaly import QUALITY_REASONS, SITE_WIDE_SHARE
from utilities.guard import require, traced

Run = Tuple[pd.Timestamp, pd.Timestamp]
_ONE = pd.Timedelta(days=1)


@traced
def day_reason(row) -> str:
    """
    #Feature: Data-quality alerts
    #Scenario: Why a single meter-day cannot be trusted
    #Complexity: Time O(1) - constant

    Args:
        row: Anything with unresolved_decrements, readings and outage_kwh.

    Returns:
        "register", "none", "catchup" or "partial".

    Raises:
        ValueError: When row is None.
    """
    require("day_reason", "row", row)
    if row.unresolved_decrements > 0:
        return "register"
    if row.readings == 0:
        return "none"
    return "catchup" if row.outage_kwh > 0 else "partial"


@traced
def episodes(days) -> List[Run]:
    """
    #Feature: Data-quality alerts
    #Scenario: Group sorted dates into runs of consecutive days
    #Complexity: Time O(n) - linear

    Args:
        days: Sorted dates.

    Returns:
        (first day, last day) per run.

    Raises:
        ValueError: When days is None.
    """
    require("episodes", "days", days)
    runs, start, prev = [], None, None
    for d in days:
        if prev is not None and (d - prev).days == 1:
            prev = d
            continue
        if start is not None:
            runs.append((start, prev))
        start = prev = d
    if start is not None:
        runs.append((start, prev))
    return runs


@traced
def describe(part: pd.DataFrame) -> Tuple[str, float, float, str]:
    """
    #Feature: Data-quality alerts
    #Scenario: Title, missing hours, catch-up kWh and what to check for one episode
    #Complexity: Time O(n) - rows in the episode

    Args:
        part: The episode's rows with reason, gap_hours and outage_kwh.

    Returns:
        (kind, missing hours, catch-up kWh, check text).

    Raises:
        ValueError: When part is None.
    """
    require("describe", "part", part)
    reasons = set(part["reason"])
    hours, catchup = float(part["gap_hours"].sum()), float(part["outage_kwh"].sum())
    if "register" in reasons:
        return ("Meter register problem", hours, catchup,
                "Check the meter's register and configuration; consumption on these days cannot be measured.")
    if reasons & {"none", "catchup"}:
        return "Meter stopped reporting", hours, catchup, "Check the meter's power, gateway or network connection."
    if hours > 0:
        return ("Incomplete readings", hours, catchup,
                "Check the meter's communication; the readings have gaps of more than an hour.")
    return ("Readings less frequent than normal", hours, catchup,
            "The meter reported all day, but only every 15-60 minutes instead of every few minutes. "
            "No time is missing, but the day falls below the normal-reading rule, so it is not used to "
            "judge consumption. Check the meter's reporting interval.")


@traced
def _carried_days(bad: pd.DataFrame, site_runs: List[Run]) -> Set[Tuple[str, pd.Timestamp]]:
    """
    #Feature: Data-quality alerts
    #Scenario: Meter-days that continue a meter's own outage across a site-wide day
    #Complexity: Time O(r*m) - site runs times meters

    A meter already in its own outage when a site-wide run starts keeps those
    site days in its own episode (not counted again in the site row).

    Args:
        bad: Untrusted meter-days with outage_at_day_start.
        site_runs: Site-wide runs.

    Returns:
        Set of (meter label, date).

    Raises:
        ValueError: When an argument is None.
    """
    require("_carried_days", "bad", bad)
    require("_carried_days", "site_runs", site_runs)
    bad_keys = set(zip(bad["meter_label"], bad["date"]))
    carried: Set[Tuple[str, pd.Timestamp]] = set()
    for start, end in site_runs:
        for r in bad[bad["date"] == start].itertuples():
            if (r.meter_label, start - _ONE) in bad_keys and bool(r.outage_at_day_start):
                day = start
                while day <= end and (r.meter_label, day) in bad_keys:
                    carried.add((r.meter_label, day))
                    day += _ONE
    return carried


@traced
def _handover_days(frame: pd.DataFrame) -> List[pd.Timestamp]:
    """
    #Feature: Data-quality alerts
    #Scenario: The days the readings switch from the database export to the Excel export
    #Complexity: Time O(d) - days

    Args:
        frame: The meter-day frame with a source column.

    Returns:
        Last database day of each switch.

    Raises:
        ValueError: When frame is None.
    """
    require("_handover_days", "frame", frame)
    sources = frame.groupby("date")["source"].agg(lambda s: set(s))
    return [d for d in sources.index if "database" in sources[d]
            and (d + _ONE) in sources.index and "excel" in sources[d + _ONE]]


@traced
def _site_rows(bad: pd.DataFrame, site_runs: List[Run], handover: List[pd.Timestamp]) -> List[Dict]:
    """
    #Feature: Data-quality alerts
    #Scenario: One row per site-wide outage or data handover gap
    #Complexity: Time O(r) - site runs

    Args:
        bad: Untrusted meter-days with a carried flag.
        site_runs: Site-wide runs.
        handover: Output of _handover_days.

    Returns:
        Alert rows.

    Raises:
        ValueError: When an argument is None.
    """
    require("_site_rows", "bad", bad)
    require("_site_rows", "site_runs", site_runs)
    rows = []
    for start, end in site_runs:
        part = bad[bad["date"].between(start, end) & ~bad["carried"]]
        _, hours, catchup, _ = describe(part)
        at_handover = any(start <= d <= end for d in handover)
        rows.append({"kind": "quality", "scope": "site", "meter_label": "", "zone": 0, "start": start, "end": end,
                     "days": (end - start).days + 1, "meters": int(part["meter_label"].nunique()),
                     "title": "Data handover gap" if at_handover else "Site-wide data outage",
                     "missing_hours": round(hours, 1), "catchup_kwh": round(catchup, 1),
                     "check": ("This is where the database export ends and the Excel export begins; the hours "
                               "in between are in neither file. Not a meter fault." if at_handover else
                               "Many meters stopped together: check the site gateway, network or data platform, "
                               "not the meters one by one.")})
    return rows


@traced
def _meter_rows(bad: pd.DataFrame, site_runs: List[Run]) -> List[Dict]:
    """
    #Feature: Data-quality alerts
    #Scenario: One row per meter-level episode of untrusted days
    #Complexity: Time O(m*d) - meters times days

    Args:
        bad: Untrusted meter-days with a carried flag.
        site_runs: Site-wide runs.

    Returns:
        Alert rows.

    Raises:
        ValueError: When an argument is None.
    """
    require("_meter_rows", "bad", bad)
    require("_meter_rows", "site_runs", site_runs)

    def in_site(d):
        return any(s <= d <= e for s, e in site_runs)

    rows = []
    for meter, part in bad.groupby("meter_label"):
        own = part[~part["date"].map(in_site) | part["carried"]]
        for start, end in episodes(sorted(own["date"])):
            run = own[own["date"].between(start, end)]
            kind, hours, catchup, check = describe(run)
            rows.append({"kind": "quality", "scope": "meter", "meter_label": meter, "zone": int(run["zone"].iloc[0]),
                         "start": start, "end": end, "days": (end - start).days + 1, "meters": 1, "title": kind,
                         "missing_hours": round(hours, 1), "catchup_kwh": round(catchup, 1), "check": check})
    return rows


@traced
def quality_alerts(frame: pd.DataFrame) -> pd.DataFrame:
    """
    #Feature: Data-quality alerts
    #Scenario: Episodes of untrustworthy days (meter-level and site-wide)
    #Complexity: Time O(n) - meter-days

    Args:
        frame: Scored meter-day frame with eligible, date, source and quality columns.

    Returns:
        One row per episode, sorted by start; empty when every day is trusted.

    Raises:
        ValueError: When frame is None.
    """
    require("quality_alerts", "frame", frame)
    bad = frame[~frame["eligible"]].copy()
    if bad.empty:
        return pd.DataFrame()
    bad["reason"] = [day_reason(r) for r in bad.itertuples()]
    per_day = bad.groupby("date")["meter_label"].nunique()
    site_days = sorted(per_day[per_day >= SITE_WIDE_SHARE * frame["meter_label"].nunique()].index)
    site_runs = episodes(site_days)
    carried = _carried_days(bad, site_runs)
    bad["carried"] = [key in carried for key in zip(bad["meter_label"], bad["date"])]
    rows = _site_rows(bad, site_runs, _handover_days(frame)) + _meter_rows(bad, site_runs)
    return pd.DataFrame(rows).sort_values(["start", "scope", "meter_label"]).reset_index(drop=True)


@traced
def historical_register_check(daily: pd.DataFrame, meter: str, day: str) -> Dict:
    """
    #Feature: Data-quality alerts
    #Scenario: A known earlier register jump predates the forecasts; check the quality logic flags it
    #Complexity: Time O(n) - rows

    Args:
        daily: The daily table.
        meter: Meter label of the known jump.
        day: Date of the known jump (YYYY-MM-DD).

    Returns:
        Dict with found, trusted_day, reason and flagged_as_data_problem.

    Raises:
        ValueError: When an argument is None.
    """
    require("historical_register_check", "daily", daily)
    require("historical_register_check", "meter", meter)
    row = daily[(daily["meter_label"] == meter) & (daily["date"] == pd.Timestamp(day))]
    if row.empty:
        return {"found": False}
    r = row.iloc[0]
    return {"found": True, "meter": meter, "date": day, "trusted_day": bool(r["measurable_day"]),
            "reason": QUALITY_REASONS[day_reason(r)] if not r["measurable_day"] else "none (day trusted)",
            "flagged_as_data_problem": bool(not r["measurable_day"])}
