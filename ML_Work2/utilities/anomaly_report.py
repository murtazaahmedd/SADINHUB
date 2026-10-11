"""Evaluation of the detectors and the data block for the dashboard's Alerts tab.

There is no record of confirmed faults, so no "accuracy %" is claimed. What is measured instead:
  * alerts per week in each period (can staff review them?),
  * in the untouched final test: where the two detectors agree, what only the
    simple rule flags and what only the ML model flags,
  * known events from the data review: the first alert in the event's direction
    and what each method flagged during the event's first week,
  * a synthetic check: each checked final-test meter-day is changed on its own to
    60% above (or below) its real reading, with its history and that morning's
    forecast left as they were. Only flags in the same direction as the change
    count. This only proves the system reacts to a sudden change; it says nothing
    about how well real equipment problems are found.
"""
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from constants.anomaly import (CHECK_TEXT, EVENT_LEAD_DAYS, EVENT_WEEK_DAYS, FLOOR_KWH,
                               MIN_DAYS_FOR_WEEKLY_RATE)
from constants.known_events import KNOWN_EVENTS, KNOWN_SITE_EVENTS
from utilities.anomaly_quality import day_reason
from utilities.guard import require, traced

Period = Tuple[str, str, str]


@traced
def _checked(period: str, frame: pd.DataFrame) -> pd.Series:
    """
    #Feature: Alert evaluation
    #Scenario: Days of a period that were actually scored
    #Complexity: Time O(n) - rows

    Args:
        period: Period name.
        frame: Scored rows.

    Returns:
        Boolean series.

    Raises:
        ValueError: When an argument is None.
    """
    require("_checked", "period", period)
    require("_checked", "frame", frame)
    return frame["period"].eq(period) & frame["eligible"] & frame["history_ok"]


@traced
def _days(period: str, frame: pd.DataFrame) -> int:
    """
    #Feature: Alert evaluation
    #Scenario: Calendar days from the period's first checked day (the warm-up is excluded)
    #Complexity: Time O(n) - rows

    Args:
        period: Period name.
        frame: Scored rows.

    Returns:
        Number of days, 0 when nothing was checked.

    Raises:
        ValueError: When an argument is None.
    """
    require("_days", "period", period)
    require("_days", "frame", frame)
    part = frame[frame["period"] == period]
    checked = part.loc[_checked(period, frame).loc[part.index], "date"]
    if checked.empty:
        return 0
    return int(part.loc[part["date"] >= checked.min(), "date"].nunique())


@traced
def period_rows(scored: pd.DataFrame, alerts: pd.DataFrame, quality: pd.DataFrame, plan: List[Period]) -> List[Dict]:
    """
    #Feature: Alert evaluation
    #Scenario: Alerts per week in each period
    #Complexity: Time O(p*n) - periods times rows

    Args:
        scored: Scored rows.
        alerts: Rows with a detector flag.
        quality: Data-problem episodes.
        plan: Period plan.

    Returns:
        One dict per period that has checked days.

    Raises:
        ValueError: When an argument is None.
    """
    require("period_rows", "scored", scored)
    require("period_rows", "alerts", alerts)
    rows = []
    for name, start, end in plan:
        part = alerts[alerts["period"] == name]
        days = _days(name, scored)
        if not days:
            continue
        weeks = days / 7.0
        rate = (lambda n: round(float(n) / weeks, 1)) if days >= MIN_DAYS_FOR_WEEKLY_RATE else (lambda n: None)
        q = quality[(quality["start"] <= pd.Timestamp(end)) & (quality["end"] >= pd.Timestamp(start))] \
            if len(quality) else quality
        checked = _checked(name, scored)
        rows.append({
            "period": name, "from": str(scored.loc[checked, "date"].min().date()),
            "to": str(scored.loc[scored["period"] == name, "date"].max().date()),
            "days": days, "weeks": round(weeks, 1), "days_checked": int(checked.sum()),
            "rule_alerts": int(part["rule_flag"].sum()), "ml_alerts": int(part["ml_flag"].sum()),
            "either_alerts": int(len(part)), "rule_per_week": rate(part["rule_flag"].sum()),
            "ml_per_week": rate(part["ml_flag"].sum()), "either_per_week": rate(len(part)),
            "higher": int((part["direction"] == "higher").sum()), "lower": int((part["direction"] == "lower").sum()),
            "data_problem_episodes": int(len(q))})
    return rows


@traced
def event_rows(alerts: pd.DataFrame) -> List[Dict]:
    """
    #Feature: Alert evaluation
    #Scenario: What each detector did around each known event
    #Complexity: Time O(e*n) - events times alerts

    Args:
        alerts: Rows with a detector flag.

    Returns:
        One dict per known event.

    Raises:
        ValueError: When alerts is None.
    """
    require("event_rows", "alerts", alerts)
    events = []
    for meter, month, direction, text in KNOWN_EVENTS:
        month_start = pd.Timestamp(month + "-01")
        window = alerts[(alerts["meter_label"] == meter) & (alerts["direction"] == direction)
                        & alerts["date"].between(month_start - pd.Timedelta(days=EVENT_LEAD_DAYS),
                                                 month_start + pd.offsets.MonthEnd(0))]
        if window.empty:
            events.append({"event": text, "meter": meter, "month": month, "direction": direction, "period": None,
                           "alerts": 0, "rule": 0, "ml": 0, "first_alert": None, "first_alert_by": None})
            continue
        first = window["date"].min()
        week = window[window["date"] < first + pd.Timedelta(days=EVENT_WEEK_DAYS)]
        first_row = window[window["date"] == first].iloc[0]
        events.append({"event": text, "meter": meter, "month": month, "direction": direction,
                       "period": first_row["period"], "first_alert": str(first.date()),
                       "first_alert_by": first_row["detector"], "alerts": int(len(week)),
                       "rule": int(week["rule_flag"].sum()), "ml": int(week["ml_flag"].sum())})
    return events


@traced
def site_event_rows(quality: pd.DataFrame, plan: List[Period]) -> List[Dict]:
    """
    #Feature: Alert evaluation
    #Scenario: Known site-wide outages must be reported as data problems, not low consumption
    #Complexity: Time O(e*q) - events times episodes

    Args:
        quality: Data-problem episodes.
        plan: Period plan.

    Returns:
        One dict per known site event.

    Raises:
        ValueError: When quality is None.
    """
    require("site_event_rows", "quality", quality)
    site = quality[quality["scope"] == "site"] if len(quality) else quality
    rows = []
    for month, text in KNOWN_SITE_EVENTS:
        first, last = pd.Timestamp(month + "-01"), pd.Timestamp(month + "-01") + pd.offsets.MonthEnd(0)
        hit = site[(site["start"] <= last) & (site["end"] >= first)] if len(site) else site
        period = next((n for n, s, e in plan if s <= str(first.date()) <= e), None)
        rows.append({"event": text, "meter": "all meters", "month": month, "direction": None, "period": period,
                     "alerts": int(len(hit)), "rule": None, "ml": None,
                     "first_alert": str(hit["start"].min().date()) if len(hit) else None, "first_alert_by": None,
                     "note": "reported as one data problem, not as low consumption"})
    return rows


@traced
def _check_text(row) -> str:
    """
    #Feature: Alert evaluation
    #Scenario: What a person should check for one alert
    #Complexity: Time O(1) - constant

    Args:
        row: Alert row with direction, levels and kWh references.

    Returns:
        Text from constants.anomaly.CHECK_TEXT.

    Raises:
        ValueError: When row is None.
    """
    require("_check_text", "row", row)
    quiet_before = not np.isfinite(row.recent_level_kwh) or row.recent_level_kwh < FLOOR_KWH
    if row.direction == "higher" and row.expected_kwh < row.floor_kwh and quiet_before:
        return CHECK_TEXT["switch_on"]
    if (row.direction == "higher" and not quiet_before and np.isfinite(row.previous_kwh)
            and row.previous_kwh < row.floor_kwh and row.actual_kwh <= 1.25 * row.recent_level_kwh):
        return CHECK_TEXT["reopened"]
    if row.direction == "lower" and np.isfinite(row.recent_level_kwh) and row.actual_kwh < 0.1 * row.recent_level_kwh:
        return CHECK_TEXT["near_zero"]
    return CHECK_TEXT[row.direction]


@traced
def alert_table(scored: pd.DataFrame) -> pd.DataFrame:
    """
    #Feature: Alert evaluation
    #Scenario: Consumption alerts (either detector) as plain rows, largest kWh difference first
    #Complexity: Time O(n log n) - sort

    Args:
        scored: Scored rows.

    Returns:
        The alert table.

    Raises:
        ValueError: When scored is None.
    """
    require("alert_table", "scored", scored)
    a = scored[scored["detector"] != ""].copy()
    a["check"] = [_check_text(r) for r in a.itertuples()]
    a["strength"] = np.where(a["diff_kwh"].abs() >= 0.5 * a["denominator_kwh"], "much ", "")
    cols = ["meter_label", "zone", "date", "period", "direction", "strength", "detector", "actual_kwh",
            "expected_kwh", "diff_kwh", "pct_of_expected", "usual_error_kwh", "z", "ml_score",
            "recent_level_kwh", "previous_kwh", "previous_date", "similar_ref_kwh", "jump", "similar", "level", "check"]
    a = a[cols].copy()
    a["abs_diff"] = a["diff_kwh"].abs()
    return a.sort_values(["date", "abs_diff"], ascending=[False, False]).drop(columns="abs_diff").reset_index(drop=True)


@traced
def _num(v, digits: int = 1):
    """
    #Feature: Alert evaluation
    #Scenario: JSON-safe rounded number
    #Complexity: Time O(1) - constant

    Args:
        v: A number or None.
        digits: Decimal places.

    Returns:
        A rounded float or None.

    Raises:
        None.
    """
    return None if v is None or not np.isfinite(v) else round(float(v), digits)


@traced
def _meter_series(scored: pd.DataFrame, dates: list) -> Dict:
    """
    #Feature: Alert evaluation
    #Scenario: Actual, expected and data-quality code of every day, per meter
    #Complexity: Time O(n) - rows

    Args:
        scored: Scored rows.
        dates: Sorted unique dates.

    Returns:
        meter -> {actual, expected, quality}.

    Raises:
        ValueError: When scored is None.
    """
    require("_meter_series", "scored", scored)
    day_index = {d: i for i, d in enumerate(dates)}
    codes = {"register": "r", "none": "n", "catchup": "c", "partial": "p"}
    meters = {}
    for meter, part in scored.groupby("meter_label"):
        actual, expected, quality_codes = [None] * len(dates), [None] * len(dates), ["."] * len(dates)
        for r in part.itertuples():
            i = day_index[r.date]
            expected[i] = round(float(r.expected_kwh), 1)
            if r.eligible:
                actual[i] = round(float(r.actual_kwh), 1)
            else:
                quality_codes[i] = codes[day_reason(r)]
        meters[meter] = {"actual": actual, "expected": expected, "quality": "".join(quality_codes)}
    return meters


@traced
def _alert_rows(alerts: pd.DataFrame) -> List[Dict]:
    """
    #Feature: Alert evaluation
    #Scenario: Consumption alerts as JSON-ready dicts
    #Complexity: Time O(n) - alerts

    Args:
        alerts: Output of alert_table.

    Returns:
        One dict per alert.

    Raises:
        ValueError: When alerts is None.
    """
    require("_alert_rows", "alerts", alerts)
    return [{"meter": r.meter_label, "zone": int(r.zone), "date": str(pd.Timestamp(r.date).date()),
             "period": r.period, "direction": r.direction, "much": bool(r.strength), "detector": r.detector,
             "actual": _num(r.actual_kwh), "expected": _num(r.expected_kwh), "diff": _num(r.diff_kwh),
             "pct": _num(r.pct_of_expected * 100 if np.isfinite(r.pct_of_expected) else np.nan, 0),
             "usual": _num(r.usual_error_kwh), "z": _num(r.z, 2), "check": r.check,
             "level_kwh": _num(r.recent_level_kwh), "prev_kwh": _num(r.previous_kwh),
             "prev_date": str(pd.Timestamp(r.previous_date).date()) if pd.notna(r.previous_date) else None,
             "similar_kwh": _num(r.similar_ref_kwh), "f_jump": _num(r.jump, 2), "f_similar": _num(r.similar, 2),
             "f_level": _num(r.level, 2)} for r in alerts.itertuples()]


@traced
def dashboard_block(scored: pd.DataFrame, alerts: pd.DataFrame, quality: pd.DataFrame, evaluation: Dict,
                    inputs: Dict, extras: Dict) -> Dict:
    """
    #Feature: Alert evaluation
    #Scenario: Everything the Alerts tab needs (plain JSON-ready values)
    #Complexity: Time O(n) - rows

    Args:
        scored: Scored rows.
        alerts: Output of alert_table.
        quality: Data-problem episodes.
        evaluation: Output of evaluate.
        inputs: Fingerprints of the files the alerts were computed from.
        extras: incidents and night alerts as JSON-ready lists.

    Returns:
        The Alerts tab data.

    Raises:
        ValueError: When an argument is None.
    """
    require("dashboard_block", "scored", scored)
    require("dashboard_block", "alerts", alerts)
    dates = sorted(scored["date"].unique())
    quality_rows = [{"scope": r.scope, "meter": r.meter_label, "zone": int(r.zone),
                     "start": str(pd.Timestamp(r.start).date()), "end": str(pd.Timestamp(r.end).date()),
                     "days": int(r.days), "meters": int(r.meters), "title": r.title,
                     "missing_hours": float(r.missing_hours), "catchup_kwh": float(r.catchup_kwh), "check": r.check}
                    for r in quality.itertuples()]
    return {"meta": {"inputs": inputs, "first_day": str(pd.Timestamp(dates[0]).date()),
                     "last_day": str(pd.Timestamp(dates[-1]).date())},
            "dates": [str(pd.Timestamp(d).date()) for d in dates],
            "meters": _meter_series(scored, dates), "alerts": _alert_rows(alerts), "quality": quality_rows,
            "evaluation": evaluation, **extras}


def _day(value) -> str:
    """ISO date text of a timestamp."""
    return str(pd.Timestamp(value).date())


@traced
def incident_rows(incidents: pd.DataFrame) -> List[Dict]:
    """
    #Feature: Alert evaluation
    #Scenario: Incidents as JSON-ready dicts
    #Complexity: Time O(n) - incidents

    Args:
        incidents: Output of build_incidents.

    Returns:
        One dict per incident.

    Raises:
        ValueError: When incidents is None.
    """
    require("incident_rows", "incidents", incidents)
    return [{"meter": r.meter_label, "zone": int(r.zone), "direction": r.direction, "start": _day(r.start),
             "end": _day(r.end), "span": int(r.span_days), "alert_days": int(r.alert_days),
             "kwh": _num(r.total_diff_kwh), "peak": _num(r.peak_diff_kwh), "z": _num(r.max_abs_z, 2),
             "detectors": r.detectors, "sar": _num(r.impact_sar, 0), "active": bool(r.active), "period": r.period,
             "check": r.check} for r in incidents.itertuples()]


@traced
def night_rows(night: pd.DataFrame) -> List[Dict]:
    """
    #Feature: Alert evaluation
    #Scenario: Night base-load alerts as JSON-ready dicts
    #Complexity: Time O(n) - alerts

    Args:
        night: Output of AnomalyService night alerts.

    Returns:
        One dict per alert.

    Raises:
        ValueError: When night is None.
    """
    require("night_rows", "night", night)
    return [{"meter": r.meter_label, "zone": int(r.zone), "date": _day(r.date), "period": r.period,
             "kw": _num(r.night_kw, 2), "usual_kw": _num(r.base_kw, 2), "rise_kw": _num(r.rise_kw, 2),
             "z": _num(r.night_z, 1)} for r in night.itertuples()]


@traced
def extras_summary(alerts: pd.DataFrame, incidents: pd.DataFrame, night: pd.DataFrame, night_checked: pd.Series,
                   plan: List[Period]) -> Dict:
    """
    #Feature: Alert evaluation
    #Scenario: Counts that show what incidents and night alerts add
    #Complexity: Time O(n) - rows

    Args:
        alerts: Daily consumption alerts.
        incidents: Output of build_incidents.
        night: Night base-load alerts with a period column.
        night_checked: Number of scored nights per period.
        plan: Period plan.

    Returns:
        Dict with incident and night-load counts per period.

    Raises:
        ValueError: When an argument is None.
    """
    require("extras_summary", "alerts", alerts)
    require("extras_summary", "incidents", incidents)
    per = {}
    for name, start, end in plan:
        n_inc = int(((incidents["start"] >= start) & (incidents["start"] <= end)).sum()) if len(incidents) else 0
        n_day = int(((alerts["date"] >= start) & (alerts["date"] <= end)).sum()) if len(alerts) else 0
        n_night = int((night["period"] == name).sum()) if len(night) else 0
        per[name] = {"daily_alerts": n_day, "incidents": n_inc, "night_alerts": n_night,
                     "nights_checked": int(night_checked.get(name, 0))}
    return {"per_period": per, "incidents_total": int(len(incidents)), "incidents_active": int(incidents["active"].sum())
            if len(incidents) else 0, "night_total": int(len(night))}
