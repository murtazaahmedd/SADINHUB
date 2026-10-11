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

Missing time is measured only from reading gaps longer than an hour; a meter
that simply reported every 15-60 minutes has no missing time.

Consecutive problem days of one meter form one episode. Days on which at least
half of all meters had a problem are reported once as a site-wide outage; a
meter whose own outage was already running when the site-wide day began stays
in its own episode (one continuous outage is never split in two).
"""
import pandas as pd

SITE_WIDE_SHARE = 0.5

REASONS = {
    "register": "register problem",
    "none": "no readings",
    "catchup": "catch-up booked after an outage",
    "partial": "incomplete readings",
}


def day_reason(row):
    if row.unresolved_decrements > 0:
        return "register"
    if row.readings == 0:
        return "none"
    if row.outage_kwh > 0:
        return "catchup"
    return "partial"


def _episodes(days):
    """Group sorted dates into runs of consecutive days."""
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


def _describe(part):
    reasons = set(part["reason"])
    hours = float(part["gap_hours"].sum())
    catchup = float(part["outage_kwh"].sum())
    if "register" in reasons:
        kind = "Meter register problem"
        check = "Check the meter's register and configuration; consumption on these days cannot be measured."
    elif reasons & {"none", "catchup"}:
        kind = "Meter stopped reporting"
        check = "Check the meter's power, gateway or network connection."
    elif hours > 0:
        kind = "Incomplete readings"
        check = "Check the meter's communication; the readings have gaps of more than an hour."
    else:
        kind = "Readings less frequent than normal"
        check = ("The meter reported all day, but only every 15-60 minutes instead of every few minutes. "
                 "No time is missing, but the day falls below the normal-reading rule, so it is not used to "
                 "judge consumption. Check the meter's reporting interval.")
    return kind, hours, catchup, check


def quality_alerts(frame):
    """Episodes of untrustworthy days (meter-level and site-wide)."""
    bad = frame[~frame["eligible"]].copy()
    if bad.empty:
        return pd.DataFrame()
    bad["reason"] = [day_reason(r) for r in bad.itertuples()]
    meters_total = frame["meter_label"].nunique()
    per_day = bad.groupby("date")["meter_label"].nunique()
    site_days = sorted(per_day[per_day >= SITE_WIDE_SHARE * meters_total].index)
    site_runs = _episodes(site_days)

    # A meter already in its own outage when a site-wide run starts keeps those
    # site days in its own episode (not counted again in the site row).
    one = pd.Timedelta(days=1)
    bad_keys = set(zip(bad["meter_label"], bad["date"]))
    carried = set()
    for start, end in site_runs:
        for r in bad[bad["date"] == start].itertuples():
            if (r.meter_label, start - one) in bad_keys and bool(r.outage_at_day_start):
                day = start
                while day <= end and (r.meter_label, day) in bad_keys:
                    carried.add((r.meter_label, day))
                    day += one
            # carried meter-days continue that meter's own outage
    bad["carried"] = [key in carried for key in zip(bad["meter_label"], bad["date"])]

    # The day the readings switch from the database export to the Excel export.
    sources = frame.groupby("date")["source"].agg(lambda s: set(s))
    handover = [d for d in sources.index if "database" in sources[d]
                and (d + one) in sources.index and "excel" in sources[d + one]]

    rows = []
    for start, end in site_runs:
        part = bad[bad["date"].between(start, end) & ~bad["carried"]]
        kind, hours, catchup, _ = _describe(part)
        at_handover = any(start <= d <= end for d in handover)
        rows.append({"kind": "quality", "scope": "site", "meter_label": "", "zone": 0,
                     "start": start, "end": end, "days": (end - start).days + 1,
                     "meters": int(part["meter_label"].nunique()),
                     "title": "Data handover gap" if at_handover else "Site-wide data outage",
                     "missing_hours": round(hours, 1), "catchup_kwh": round(catchup, 1),
                     "check": ("This is where the database export ends and the Excel export begins; the hours "
                               "in between are in neither file. Not a meter fault." if at_handover else
                               "Many meters stopped together: check the site gateway, network or data platform, "
                               "not the meters one by one.")})

    def in_site(d):
        return any(s <= d <= e for s, e in site_runs)

    for meter, part in bad.groupby("meter_label"):
        own = part[~part["date"].map(in_site) | part["carried"]]
        for start, end in _episodes(sorted(own["date"])):
            run = own[own["date"].between(start, end)]
            kind, hours, catchup, check = _describe(run)
            rows.append({"kind": "quality", "scope": "meter", "meter_label": meter, "zone": int(run["zone"].iloc[0]),
                         "start": start, "end": end, "days": (end - start).days + 1, "meters": 1, "title": kind,
                         "missing_hours": round(hours, 1), "catchup_kwh": round(catchup, 1), "check": check})
    return pd.DataFrame(rows).sort_values(["start", "scope", "meter_label"]).reset_index(drop=True)


def historical_register_check(daily, meter="ZN.13-PB.1_child__1_ch3pm", date="2025-02-05"):
    """The Feb 2025 ZN.13 register jump predates the 2026 forecasts; check the quality logic flags it."""
    row = daily[(daily["meter_label"] == meter) & (daily["date"] == pd.Timestamp(date))]
    if row.empty:
        return {"found": False}
    r = row.iloc[0]
    return {"found": True, "meter": meter, "date": date, "trusted_day": bool(r["measurable_day"]),
            "reason": REASONS[day_reason(r)] if not r["measurable_day"] else "none (day trusted)",
            "flagged_as_data_problem": bool(not r["measurable_day"])}
