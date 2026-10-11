"""Group a meter's repeated alert days into incidents and rank them by impact."""
from typing import List

import pandas as pd

from constants.anomaly import INCIDENT_ACTIVE_DAYS, INCIDENT_MAX_GAP_DAYS
from constants.tariff import RATE_UP_TO_LIMIT
from utilities.guard import require, traced

_COLUMNS = ["meter_label", "zone", "direction", "start", "end", "span_days", "alert_days", "total_diff_kwh",
            "peak_diff_kwh", "max_abs_z", "detectors", "impact_sar", "active", "period", "check"]


@traced
def _runs(dates: List[pd.Timestamp]) -> List[List[pd.Timestamp]]:
    """
    #Feature: Incidents
    #Scenario: Split sorted alert dates where the quiet gap is longer than allowed
    #Complexity: Time O(n) - linear

    Args:
        dates: Sorted alert dates of one meter and direction.

    Returns:
        Lists of dates, one per incident.

    Raises:
        ValueError: When dates is None.
    """
    require("_runs", "dates", dates)
    runs: List[List[pd.Timestamp]] = []
    for d in dates:
        if runs and (d - runs[-1][-1]).days <= INCIDENT_MAX_GAP_DAYS:
            runs[-1].append(d)
        else:
            runs.append([d])
    return runs


@traced
def build_incidents(alerts: pd.DataFrame, last_day: pd.Timestamp) -> pd.DataFrame:
    """
    #Feature: Incidents
    #Scenario: One row per incident with its size, duration and money impact, biggest first
    #Complexity: Time O(n log n) - sort

    Impact is the total kWh difference on the alert days at the lower tariff rate, so it is a conservative
    estimate of the cost of the excess (or the value of the missing use), not a bill.

    Args:
        alerts: Output of alert_table (meter_label, zone, date, period, direction, detector, diff_kwh, z, check).
        last_day: Last day of data.

    Returns:
        Columns meter_label, zone, direction, start, end, span_days, alert_days, total_diff_kwh, peak_diff_kwh,
        max_abs_z, detectors, impact_sar, active, period and check; empty when there are no alerts.

    Raises:
        ValueError: When an argument is None.
    """
    require("build_incidents", "alerts", alerts)
    require("build_incidents", "last_day", last_day)
    rows = []
    for (meter, direction), g in alerts.sort_values("date").groupby(["meter_label", "direction"]):
        by_date = g.set_index("date")
        for run in _runs(list(by_date.index)):
            part = by_date.loc[run]
            top = part.loc[part["diff_kwh"].abs().idxmax()] if len(part) > 1 else part.iloc[0]
            total = float(part["diff_kwh"].sum())
            rows.append({"meter_label": meter, "zone": int(part["zone"].iloc[0]), "direction": direction,
                         "start": run[0], "end": run[-1], "span_days": (run[-1] - run[0]).days + 1,
                         "alert_days": len(run), "total_diff_kwh": round(total, 1),
                         "peak_diff_kwh": round(float(top["diff_kwh"]), 1),
                         "max_abs_z": round(float(part["z"].abs().max()), 2) if part["z"].notna().any() else None,
                         "detectors": "+".join(sorted(set(part["detector"]))),
                         "impact_sar": round(abs(total) * RATE_UP_TO_LIMIT, 2),
                         "active": bool((last_day - run[-1]).days <= INCIDENT_ACTIVE_DAYS),
                         "period": part["period"].iloc[0], "check": top["check"]})
    out = pd.DataFrame(rows, columns=_COLUMNS)
    return out.sort_values("impact_sar", ascending=False).reset_index(drop=True)
