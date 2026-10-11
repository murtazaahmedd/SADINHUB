"""Bill forecast: kWh -> SAR for each meter and each 30-day billing cycle.

Rules (given by the facility, commercial tariff):
  * Billing cycles are 30 days, counted from 1 January 2026
    (1-30 Jan, 31 Jan-1 Mar, 2-31 Mar, ...).
  * Energy charge per meter per cycle: the first 6,000 kWh at 0.22 SAR/kWh,
    every kWh above 6,000 at 0.32 SAR/kWh.
  * Plus a fixed monthly service fee set by the meter's circuit-breaker size
    (silver.dim_meter_cb_rating, matched by energy_meter_key).
  * Bill = (energy charge + service fee) x 1.15 (15% VAT).

Which kWh is used:
  * completed cycles - the forecast made on the cycle's first morning (and,
    alongside, the bill on the actual metered kWh);
  * the current cycle - actual kWh so far plus the latest forecast for the rest;
  * future cycles     - the latest forecast.
"""
from datetime import date, timedelta

import numpy as np
import pandas as pd

from . import config
from .database import connect

# ---- tariff (edit here if the tariff changes) ----
TIER_LIMIT_KWH = 6000.0
RATE_UP_TO_LIMIT = 0.22      # SAR per kWh, commercial, first 6,000 kWh
RATE_ABOVE_LIMIT = 0.32      # SAR per kWh, commercial, above 6,000 kWh
VAT_RATE = 0.15
CYCLE_DAYS = 30
CYCLE_YEAR_START = date(2026, 1, 1)

# (lowest amps inclusive, highest amps exclusive, class, monthly fee SAR)
SERVICE_FEE_CLASSES = [
    (1, 100, "Class I", 10.0),
    (100, 200, "Class II", 15.0),
    (200, 300, "Class III", 21.0),
    (300, 400, "Class IV", 22.0),
    (400, 400.000001, "Class V", 25.0),          # exactly 400 A
    (400.000001, float("inf"), "Class VI", 30.0),  # more than 400 A
]

BREAKERS_PATH = config.DATA_DIR / "meter_breakers.csv"
BILLS_PATH = config.FORECAST_DIR / "bill_forecast_2026_by_meter.csv"


def service_class(amps):
    for low, high, name, fee in SERVICE_FEE_CLASSES:
        if low <= amps < high:
            return name, fee
    raise ValueError(f"Breaker rating {amps} A is outside every service-fee class")


def bill(kwh, fee):
    """All parts of one meter's bill for one cycle."""
    kwh = max(float(kwh), 0.0)
    low = min(kwh, TIER_LIMIT_KWH)
    high = max(kwh - TIER_LIMIT_KWH, 0.0)
    # Every money amount is rounded to the halala, and the bill is the exact sum
    # of its rounded parts, so all totals on the dashboard add up.
    energy_low = round(low * RATE_UP_TO_LIMIT, 2)
    energy_high = round(high * RATE_ABOVE_LIMIT, 2)
    fee = round(float(fee), 2)
    vat = round((energy_low + energy_high + fee) * VAT_RATE, 2)
    return {"kwh": kwh, "kwh_at_022": low, "kwh_at_032": high, "energy_022_sar": energy_low,
            "energy_032_sar": energy_high, "service_fee_sar": fee, "vat_sar": vat,
            "bill_sar": round(energy_low + energy_high + fee + vat, 2)}


def cycles(last_day=config.FORECAST_END):
    """30-day cycles from 1 January; the last one may run past the forecast."""
    out, start, n = [], CYCLE_YEAR_START, 1
    last_day = pd.Timestamp(last_day).date()
    while start <= last_day:
        end = start + timedelta(days=CYCLE_DAYS - 1)
        out.append({"cycle": n, "start": start, "end": end, "full": end <= last_day})
        start, n = end + timedelta(days=1), n + 1
    return out


def load_breakers(refresh=False):
    """Breaker rating per meter, matched on energy_meter_key (never on names)."""
    if BREAKERS_PATH.exists() and not refresh:
        return pd.read_csv(BREAKERS_PATH)
    with connect() as conn, conn.cursor() as cur:
        cur.execute("""SELECT m.energy_meter_key::text, m.meter_label, cb.cb_max_rating_a
                       FROM gold.meter m
                       LEFT JOIN silver.dim_meter_cb_rating cb ON cb.energy_meter_key = m.energy_meter_key
                       ORDER BY m.meter_label""")
        rows = cur.fetchall()
    out = pd.DataFrame(rows, columns=["meter_key", "meter_label", "breaker_a"])
    missing = out[out["breaker_a"].isna()]
    if not missing.empty:
        raise RuntimeError(f"No breaker rating for: {', '.join(missing['meter_label'])}")
    out["breaker_a"] = out["breaker_a"].astype(float)
    out[["service_class", "service_fee_sar"]] = [service_class(a) for a in out["breaker_a"]]
    out.to_csv(BREAKERS_PATH, index=False)
    return out


def build_bills(panel, daily, preds, future, issue, breakers):
    """One row per meter per 30-day cycle with predicted (and actual) bill."""
    last_complete = daily.loc[daily["complete_day"], "date"].max()
    fee = breakers.set_index("meter_label")
    missing = sorted(set(panel.meters) - set(fee.index))
    if missing:
        raise RuntimeError(f"Meters without a breaker rating: {missing}")
    gold = (daily[daily["complete_day"]].pivot_table(index="meter_label", columns="date", values="gold_kwh")
            .reindex(index=panel.meters))
    fut = future.pivot_table(index="meter_label", columns="date", values="forecast_kwh").reindex(panel.meters)
    morning = preds[preds["days_ahead"] < CYCLE_DAYS]
    rows = []
    for c in cycles():
        start, end = pd.Timestamp(c["start"]), pd.Timestamp(c["end"])
        days = pd.date_range(start, min(end, pd.Timestamp(config.FORECAST_END)))
        if not c["full"]:
            status = "runs into 2027"
        elif end <= last_complete:
            status = "completed"
        elif start <= last_complete:
            status = "in progress"
        else:
            status = "upcoming"
        made_on = morning[morning["origin_idx"] == panel.index_of(start)] if status == "completed" else None
        for i, label in enumerate(panel.meters):
            if status == "completed":
                predicted = float(made_on.loc[made_on["meter_idx"] == i, "pred_kwh"].sum())
                basis = f"forecast made on {start:%d %b %Y}"
            else:
                past = [d for d in days if d <= last_complete]
                ahead = [d for d in days if d > last_complete]
                predicted = float(gold.loc[label, past].sum()) if past else 0.0
                predicted += float(fut.loc[label, ahead].sum()) if ahead else 0.0
                basis = (f"actual to {last_complete:%d %b} + forecast made {issue:%d %b %Y}" if past
                         else f"forecast made {issue:%d %b %Y}")
            actual = float(gold.loc[label, list(days)].sum()) if status == "completed" else np.nan
            p = bill(predicted, fee.loc[label, "service_fee_sar"])
            a = bill(actual, fee.loc[label, "service_fee_sar"]) if np.isfinite(actual) else {}
            row = {"cycle": c["cycle"], "cycle_start": start.date(), "cycle_end": end.date(), "status": status,
                   "days_in_2026": len(days), "meter_label": label, "zone": int(panel.zone[i]),
                   "breaker_a": fee.loc[label, "breaker_a"], "service_class": fee.loc[label, "service_class"],
                   "basis": basis}
            row.update({f"predicted_{k}": v for k, v in p.items()})
            row.update({f"actual_{k}": a.get(k, np.nan) for k in p})
            rows.append(row)
    out = pd.DataFrame(rows)
    num = out.select_dtypes("number").columns.difference(["cycle", "zone", "days_in_2026", "breaker_a"])
    out[num] = out[num].round(2)
    out.to_csv(BILLS_PATH, index=False)
    return out
