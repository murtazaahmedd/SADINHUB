"""Evaluation of the detectors and the data block for the dashboard's Alerts tab.

There is no record of confirmed faults, so no "accuracy %" is claimed. What is
measured instead:
  * alerts per week in each period (can staff review them?),
  * in the untouched final test (September): where the two detectors agree,
    what only the simple rule flags and what only the ML model flags,
  * known events from the data review: the first alert in the event's direction
    and what each method flagged during the event's first week, each reported in
    the period that first alert falls in,
  * a synthetic check: each checked September meter-day is changed on its own to
    60% above (or below) its real reading, with its history and that morning's
    forecast left as they were - a one-day jump or dip. Only flags in the same
    direction as the change count. This only proves the system reacts to a
    sudden change; it says nothing about how well real equipment problems are found.
"""
import numpy as np
import pandas as pd

from .detectors import PERIODS, rescore
from .features import FLOOR_KWH, with_actual

# Known events from the forecasting work (documentation/HOW_THE_FORECAST_WORKS.md).
# Only the meter, the month the doc names and the direction are given here; the
# onset and what the detectors did are measured. The doc's months come from
# monthly averages, so the search starts a week before that month.
KNOWN_EVENTS = [
    ("ZN.16-PB_child__1_ch3pm", "2026-06", "higher", "ZN.16 switched on after months at 0 kWh"),
    ("ZN.8-PB.2_child__1_ch3pm", "2026-07", "higher", "ZN.8-PB.2 took on a large new load"),
    ("ZN.13-PB.1_child__1_ch3pm", "2026-09", "lower", "ZN.13-PB.1 fell to a much lower level"),
]
EVENT_LEAD_DAYS = 7
EVENT_WEEK_DAYS = 7
SYNTHETIC_FACTORS = {"60% higher": 1.6, "60% lower": 0.4}
MIN_DAYS_FOR_WEEKLY_RATE = 7

CHECK = {
    "higher": "Check: equipment left running, longer operating hours, a new tenant or equipment, or a meter fault.",
    "lower": "Check: closure or holiday, vacancy, equipment switched off, or a meter fault.",
    "switch_on": "Check: the meter started using power again after a quiet spell - new tenant or equipment switched on?",
    "near_zero": "Check: consumption fell almost to zero - is the area closed or the supply off?",
    "reopened": ("Check: consumption came back to this meter's normal level after a quiet spell; the morning "
                 "forecast expected the quiet spell to continue - confirm the area reopened (no action needed if so)."),
}


def _checked(period, frame):
    return frame["period"].eq(period) & frame["eligible"] & frame["history_ok"]


def _days(period, frame):
    """Calendar days from the period's first checked day (the 2-week warm-up is excluded)."""
    part = frame[frame["period"] == period]
    checked = part.loc[_checked(period, frame).loc[part.index], "date"]
    if checked.empty:
        return 0
    return int(part.loc[part["date"] >= checked.min(), "date"].nunique())


def evaluate(scored, quality, forest, settings, register_check):
    alerts = scored[scored["detector"] != ""]
    per_period = []
    for name, start, end in PERIODS:
        part = alerts[alerts["period"] == name]
        days = _days(name, scored)
        if not days:
            continue
        weeks = days / 7.0
        rate = (lambda n: round(n / weeks, 1)) if days >= MIN_DAYS_FOR_WEEKLY_RATE else (lambda n: None)
        q = quality[(quality["start"] <= pd.Timestamp(end)) & (quality["end"] >= pd.Timestamp(start))]
        checked = _checked(name, scored)
        per_period.append({
            "period": name, "from": str(scored.loc[checked, "date"].min().date()),
            "to": str(scored.loc[scored["period"] == name, "date"].max().date()),
            "days": days, "weeks": round(weeks, 1),
            "days_checked": int(checked.sum()),
            "rule_alerts": int(part["rule_flag"].sum()), "ml_alerts": int(part["ml_flag"].sum()),
            "either_alerts": int(len(part)),
            "rule_per_week": rate(part["rule_flag"].sum()), "ml_per_week": rate(part["ml_flag"].sum()),
            "either_per_week": rate(len(part)),
            "higher": int((part["direction"] == "higher").sum()), "lower": int((part["direction"] == "lower").sum()),
            "data_problem_episodes": int(len(q)),
        })

    test = alerts[alerts["period"] == "final test"]
    agreement = {"both": int((test["detector"] == "both").sum()), "rule_only": int((test["detector"] == "rule").sum()),
                 "ml_only": int((test["detector"] == "ml").sum())}

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
                       "first_alert_by": first_row["detector"],
                       "alerts": int(len(week)), "rule": int(week["rule_flag"].sum()), "ml": int(week["ml_flag"].sum())})
    site = quality[quality["scope"] == "site"]
    feb = site[(site["start"] <= "2026-02-28") & (site["end"] >= "2026-02-01")]
    events.append({"event": "February 2026 site-wide data outage", "meter": "all meters", "month": "2026-02",
                   "direction": None, "period": "development", "alerts": int(len(feb)), "rule": None, "ml": None,
                   "first_alert": str(feb["start"].min().date()) if len(feb) else None, "first_alert_by": None,
                   "note": "reported as one data problem, not as low consumption"})

    test_rows = scored[(scored["period"] == "final test") & scored["eligible"] & scored["history_ok"]]
    synthetic = {}
    for label, factor in SYNTHETIC_FACTORS.items():
        moved = with_actual(test_rows, test_rows["actual_kwh"] * factor)
        rule, ml = rescore(moved, forest, settings["ml_cutoff"])
        right_way = np.sign(moved["diff_kwh"]) == np.sign(factor - 1)   # flag must match the change
        synthetic[label] = {"days_tested": int(len(moved)),
                            "rule": round(float((rule & right_way).mean()) * 100, 1),
                            "ml": round(float((ml & right_way).mean()) * 100, 1),
                            "either": round(float(((rule | ml) & right_way).mean()) * 100, 1),
                            "wrong_direction_flags": int(((rule | ml) & ~right_way).sum())}

    return {"periods": per_period, "final_test_agreement": agreement, "known_events": events,
            "synthetic_check": synthetic, "historical_register_check": register_check, "settings": settings}


def _check_text(row):
    quiet_before = not np.isfinite(row.recent_level_kwh) or row.recent_level_kwh < FLOOR_KWH
    if row.direction == "higher" and row.expected_kwh < row.floor_kwh and quiet_before:
        return CHECK["switch_on"]
    if (row.direction == "higher" and not quiet_before and np.isfinite(row.previous_kwh)
            and row.previous_kwh < row.floor_kwh and row.actual_kwh <= 1.25 * row.recent_level_kwh):
        return CHECK["reopened"]
    if row.direction == "lower" and np.isfinite(row.recent_level_kwh) and row.actual_kwh < 0.1 * row.recent_level_kwh:
        return CHECK["near_zero"]
    return CHECK[row.direction]


def alert_table(scored):
    """Consumption alerts (either detector) as plain rows, largest kWh difference first."""
    a = scored[scored["detector"] != ""].copy()
    a["check"] = [_check_text(r) for r in a.itertuples()]
    a["strength"] = np.where(a["diff_kwh"].abs() >= 0.5 * a["denominator_kwh"], "much ", "")
    cols = ["meter_label", "zone", "date", "period", "direction", "strength", "detector", "actual_kwh",
            "expected_kwh", "diff_kwh", "pct_of_expected", "usual_error_kwh", "z", "ml_score",
            "recent_level_kwh", "previous_kwh", "previous_date", "similar_ref_kwh", "jump", "similar", "level", "check"]
    a = a[cols].copy()
    a["abs_diff"] = a["diff_kwh"].abs()
    return a.sort_values(["date", "abs_diff"], ascending=[False, False]).drop(columns="abs_diff").reset_index(drop=True)


def dashboard_block(scored, alerts, quality, evaluation, inputs):
    """Everything the Alerts tab needs (plain JSON-ready values)."""
    dates = sorted(scored["date"].unique())
    day_index = {d: i for i, d in enumerate(dates)}
    codes = {"register": "r", "none": "n", "catchup": "c", "partial": "p"}
    from .quality import day_reason

    meters = {}
    for meter, part in scored.groupby("meter_label"):
        actual = [None] * len(dates)
        expected = [None] * len(dates)
        quality_codes = ["."] * len(dates)
        for r in part.itertuples():
            i = day_index[r.date]
            expected[i] = round(float(r.expected_kwh), 1)
            if r.eligible:
                actual[i] = round(float(r.actual_kwh), 1)
            else:
                quality_codes[i] = codes[day_reason(r)]
        meters[meter] = {"actual": actual, "expected": expected, "quality": "".join(quality_codes)}

    def num(v, digits=1):
        return None if v is None or not np.isfinite(v) else round(float(v), digits)

    alert_rows = [{"meter": r.meter_label, "zone": int(r.zone), "date": str(pd.Timestamp(r.date).date()),
                   "period": r.period, "direction": r.direction, "much": bool(r.strength), "detector": r.detector,
                   "actual": num(r.actual_kwh), "expected": num(r.expected_kwh), "diff": num(r.diff_kwh),
                   "pct": num(r.pct_of_expected * 100 if np.isfinite(r.pct_of_expected) else np.nan, 0),
                   "usual": num(r.usual_error_kwh), "z": num(r.z, 2), "check": r.check,
                   "level_kwh": num(r.recent_level_kwh), "prev_kwh": num(r.previous_kwh),
                   "prev_date": str(pd.Timestamp(r.previous_date).date()) if pd.notna(r.previous_date) else None,
                   "similar_kwh": num(r.similar_ref_kwh), "f_jump": num(r.jump, 2), "f_similar": num(r.similar, 2),
                   "f_level": num(r.level, 2)}
                  for r in alerts.itertuples()]
    quality_rows = [{"scope": r.scope, "meter": r.meter_label, "zone": int(r.zone),
                     "start": str(pd.Timestamp(r.start).date()), "end": str(pd.Timestamp(r.end).date()),
                     "days": int(r.days), "meters": int(r.meters), "title": r.title,
                     "missing_hours": float(r.missing_hours), "catchup_kwh": float(r.catchup_kwh), "check": r.check}
                    for r in quality.itertuples()]
    return {"meta": {"inputs": inputs, "first_day": str(pd.Timestamp(dates[0]).date()),
                     "last_day": str(pd.Timestamp(dates[-1]).date())},
            "dates": [str(pd.Timestamp(d).date()) for d in dates],
            "meters": meters, "alerts": alert_rows, "quality": quality_rows, "evaluation": evaluation}
