"""Build the single-file dashboard: predictions and accuracy only."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .billing import (RATE_ABOVE_LIMIT, RATE_UP_TO_LIMIT, SERVICE_FEE_CLASSES, TIER_LIMIT_KWH, VAT_RATE,
                      build_bills, load_breakers)
from .scoring import accuracy, meter_periods

TEMPLATE = Path(__file__).with_name("dashboard_template.html")
CHART_JS = Path(__file__).with_name("vendor") / "chart.umd.min.js"
OUTPUT = config.DASHBOARD_DIR / "SADIN_Energy_Forecast_Dashboard.html"
HORIZONS = ["1 day", "7 days", "30 days", "calendar month"]


def _short(label):
    return label.split("_child")[0]


def _units(panel):
    units = [{"id": "facility", "level": "facility", "name": "Facility (all zones)"}]
    for z in sorted(set(panel.zone)):
        units.append({"id": f"zone-{z}", "level": "zone", "name": f"Zone {z:02d}"})
    for label, z in sorted(zip(panel.meters, panel.zone), key=lambda p: (p[1], p[0])):
        units.append({"id": label, "level": "meter", "name": _short(label), "zone": f"Zone {z:02d}"})
    return units


def _unit_key(level, unit):
    if level == "facility":
        return "facility"
    if level == "zone":
        return f"zone-{int(unit.split()[-1])}"
    return unit


def _num(value):
    """JSON-safe number: missing stays missing, accuracy below zero shows as 0."""
    if value is None or not np.isfinite(value):
        return None
    return round(max(float(value), 0.0), 1)


def _accuracy_block(levels):
    out = {}
    levels = levels.copy()
    levels["month"] = levels["issue_date"].dt.strftime("%Y-%m")
    for (level, unit), part in levels.groupby(["level", "unit"]):
        key = _unit_key(level, unit)
        entry = {}
        for horizon in HORIZONS:
            h = part[part["horizon"] == horizon]
            if h.empty:
                continue
            months = {m: _num(accuracy(g)) for m, g in h.groupby("month")}
            entry[horizon] = {"overall": _num(accuracy(h)), "months": months, "periods": int(len(h))}
        out[key] = entry
    return out


def _calendar_month_table(levels):
    out = {}
    cm = levels[levels["horizon"] == "calendar month"]
    for (level, unit), part in cm.groupby(["level", "unit"]):
        out[_unit_key(level, unit)] = [
            {"month": d.strftime("%Y-%m"), "forecast": round(f, 1), "actual": round(a, 1), "accuracy": _num(acc)}
            for d, f, a, acc in part.sort_values("issue_date")[["issue_date", "forecast", "actual", "accuracy"]]
            .itertuples(index=False)]
    return out


def _series(panel, daily, preds, future):
    day0 = preds[preds["days_ahead"] == 0]
    first, last = int(day0["target_idx"].min()), int(day0["target_idx"].max())
    hist_dates = panel.dates[first:last + 1]
    fc = np.full((len(panel.meters), last - first + 1), np.nan)
    fc[day0["meter_idx"].to_numpy(), day0["target_idx"].to_numpy() - first] = day0["pred_kwh"].to_numpy()
    # Chart line: energy per day with outage catch-ups spread over the missing
    # days (same totals as gold; gold books a whole outage on the return day).
    complete = daily[daily["complete_day"]]
    act = (complete.pivot_table(index="meter_label", columns="date", values="spread_kwh")
           .reindex(index=panel.meters, columns=hist_dates).to_numpy())

    fut_dates = sorted(future["date"].unique())
    fut = future.pivot_table(index="meter_label", columns="date", values="forecast_kwh").reindex(panel.meters)
    fut = fut.reindex(columns=fut_dates).to_numpy()

    def pack(rows):
        a = np.nansum(act[rows], 0) if len(rows) > 1 else act[rows[0]]
        f = np.nansum(fc[rows], 0) if len(rows) > 1 else fc[rows[0]]
        incomplete = np.isnan(act[rows]).all(0)
        a = np.where(incomplete, np.nan, a)
        u = np.nansum(fut[rows], 0)
        clean = lambda v: [None if np.isnan(x) else round(float(x), 1) for x in v]  # noqa: E731
        return {"actual": clean(a), "forecast": clean(f), "future": clean(u)}

    series = {"facility": pack(list(range(len(panel.meters))))}
    for z in sorted(set(panel.zone)):
        series[f"zone-{z}"] = pack(list(np.flatnonzero(panel.zone == z)))
    for i, label in enumerate(panel.meters):
        series[label] = pack([i])
    dates = {"history": [d.strftime("%Y-%m-%d") for d in hist_dates],
             "future": [pd.Timestamp(d).strftime("%Y-%m-%d") for d in fut_dates]}
    return series, dates


def _explorer(panel, preds, history_dates):
    """Everything the date explorer needs, per meter (zones/facility are sums).

    wf[m]  - the forecast made on each morning of 2026 for that day and the next
             30 days, flattened as [morning * 31 + days_ahead]
    actual[h][m][morning], scored[h][m][morning] - the gold total of that
             period and whether it can be scored (same rule as the validation)
    """
    origins = sorted(int(t) for t in preds["origin_idx"].unique())
    origin_dates = [panel.dates[t].strftime("%Y-%m-%d") for t in origins]
    if origin_dates != list(history_dates):
        raise RuntimeError("Explorer mornings do not line up with the daily chart dates.")
    index = {t: i for i, t in enumerate(origins)}
    k = int(preds["days_ahead"].max()) + 1
    m = len(panel.meters)
    wf = np.full((m, len(origins), k), np.nan)
    wf[preds["meter_idx"].to_numpy(), preds["origin_idx"].map(index).to_numpy(),
       preds["days_ahead"].to_numpy()] = preds["pred_kwh"].to_numpy()

    def clean(values, digits=1):
        return [None if not np.isfinite(v) else round(float(v), digits) for v in values]

    periods = meter_periods(panel, preds)
    actual, scored = {}, {}
    for h, part in periods.groupby("horizon"):
        a = np.full((m, len(origins)), np.nan)
        s = np.zeros((m, len(origins)), dtype=int)
        rows = part["meter_idx"].to_numpy(), part["origin_idx"].map(index).to_numpy()
        a[rows] = part["actual"].to_numpy()
        s[rows] = part["measurable"].to_numpy().astype(int)
        actual[str(h)] = [clean(row) for row in a]
        scored[str(h)] = s.tolist()
    # "1" = the meter's readings cover that whole day (the day can be scored).
    first = origins[0]
    measured = ["".join("1" if v else "0" for v in panel.measurable[i, first:first + len(origins)])
                for i in range(m)]
    return {
        "mornings": origin_dates,
        "days": k,
        "meters": list(panel.meters),
        "measured": measured,
        "wf": [clean(wf[i].ravel()) for i in range(m)],
        "actual": actual,
        "scored": scored,
    }


def _bills_block(bills, breakers):
    """Per meter, per 30-day cycle: predicted bill parts (and bill on actual kWh)."""
    first = bills.drop_duplicates("cycle").sort_values("cycle")
    cycle_list = [{"n": int(r.cycle), "start": str(r.cycle_start), "end": str(r.cycle_end),
                   "status": r.status, "days": int(r.days_in_2026), "basis": r.basis}
                  for r in first.itertuples()]
    parts = ["kwh", "kwh_at_022", "kwh_at_032", "energy_022_sar", "energy_032_sar",
             "service_fee_sar", "vat_sar", "bill_sar"]

    def nums(row, prefix):
        values = [row[f"{prefix}_{p}"] for p in parts]
        return None if not np.isfinite(values[0]) else [round(float(v), 2) for v in values]

    meters = {}
    for label, part in bills.sort_values("cycle").groupby("meter_label"):
        b = breakers.set_index("meter_label").loc[label]
        meters[label] = {"breaker_a": float(b["breaker_a"]), "class": b["service_class"],
                         "fee": float(b["service_fee_sar"]),
                         "predicted": [nums(r, "predicted") for _, r in part.iterrows()],
                         "actual": [nums(r, "actual") for _, r in part.iterrows()]}
    return {"cycles": cycle_list, "parts": parts, "meters": meters,
            "tariff": {"limit": TIER_LIMIT_KWH, "low": RATE_UP_TO_LIMIT, "high": RATE_ABOVE_LIMIT,
                       "vat": VAT_RATE, "classes": [[c[2], c[3]] for c in SERVICE_FEE_CLASSES]}}


def _outlook(daily, future):
    """Per unit: month -> actual so far + forecast for the rest of 2026."""
    actual = daily[daily["complete_day"] & (daily["date"].dt.year == 2026)].copy()
    actual["month"] = actual["date"].dt.strftime("%Y-%m")
    fut = future.copy()
    fut["month"] = fut["date"].dt.strftime("%Y-%m")
    out = {}

    def add(key, a, f):
        months = sorted(set(a.index) | set(f.index))
        out[key] = {m: {"actual": round(float(a.get(m, 0.0)), 1), "forecast": round(float(f.get(m, 0.0)), 1)}
                    for m in months}

    add("facility", actual.groupby("month")["gold_kwh"].sum(), fut.groupby("month")["forecast_kwh"].sum())
    for z, g in actual.groupby("zone"):
        add(f"zone-{z}", g.groupby("month")["gold_kwh"].sum(),
            fut[fut["zone"] == z].groupby("month")["forecast_kwh"].sum())
    for m, g in actual.groupby("meter_label"):
        add(m, g.groupby("month")["gold_kwh"].sum(),
            fut[fut["meter_label"] == m].groupby("month")["forecast_kwh"].sum())
    return out


ALERTS_JSON = config.ROOT / "anomaly_detection" / "outputs" / "alerts_dashboard.json"
ALERT_INPUTS = {"daily_actuals": config.DATA_DIR / "daily_actuals.parquet",
                "every_morning_forecast": config.VALIDATION_DIR / "every_morning_forecast_2026.parquet"}


def _alerts_block():
    """The Alerts tab's data, written by `python -m anomaly_detection.run` (None if not run yet).

    The alerts record fingerprints of the files they were computed from; if the
    forecasting outputs have changed since, the tab says the alerts are out of date.
    """
    if not ALERTS_JSON.exists():
        return None
    block = json.loads(ALERTS_JSON.read_text(encoding="utf-8"))
    current = {}
    for name, path in ALERT_INPUTS.items():
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        current[name] = digest.hexdigest()
    block["meta"]["stale"] = block["meta"].get("inputs") != current
    return block


def build_dashboard(panel, daily, preds, levels, future, issue, long_range):
    series, dates = _series(panel, daily, preds, future)
    data = {
        "long_range": long_range,
        "meta": {
            "issued": f"{issue:%d %B %Y}",
            "data_through": f"{issue - pd.Timedelta(days=1):%d %B %Y}",
            "forecast_end": f"{config.FORECAST_END:%d %B %Y}",
            "validation_from": f"{config.VALIDATION_START:%d %B %Y}",
            "target": config.ACCURACY_TARGET,
            "issue_date": f"{issue:%Y-%m-%d}",
        },
        "units": _units(panel),
        "accuracy": _accuracy_block(levels),
        "calendar_months": _calendar_month_table(levels),
        "series": series,
        "dates": dates,
        "outlook": _outlook(daily, future),
        "explorer": _explorer(panel, preds, dates["history"]),
    }
    breakers = load_breakers()
    data["bills"] = _bills_block(build_bills(panel, daily, preds, future, issue, breakers), breakers)
    data["alerts"] = _alerts_block()
    html = TEMPLATE.read_text(encoding="utf-8")
    html = html.replace("/*__CHART_JS__*/", CHART_JS.read_text(encoding="utf-8"))
    html = html.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    OUTPUT.write_text(html, encoding="utf-8")
    return OUTPUT
