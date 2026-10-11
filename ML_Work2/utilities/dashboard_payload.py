"""Pure builders for each block of the dashboard's data (everything becomes JSON-ready values)."""
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from constants.forecasting import HORIZON_LABELS
from constants.tariff import RATE_ABOVE_LIMIT, RATE_UP_TO_LIMIT, SERVICE_FEE_CLASSES, TIER_LIMIT_KWH, VAT_RATE
from errors import Errors
from utilities.guard import require, traced
from utilities.scoring import accuracy, meter_periods

_HORIZONS = list(HORIZON_LABELS.values())
_BILL_PARTS = ["kwh", "kwh_at_022", "kwh_at_032", "energy_022_sar", "energy_032_sar",
               "service_fee_sar", "vat_sar", "bill_sar"]


@traced
def _short(label: str) -> str:
    """
    #Feature: Dashboard data
    #Scenario: Short meter name without the device suffix
    #Complexity: Time O(len(label)) - linear

    Args:
        label: Full meter label.

    Returns:
        The label up to "_child".

    Raises:
        ValueError: When label is None.
    """
    require("_short", "label", label)
    return label.split("_child")[0]


@traced
def units(panel) -> List[Dict]:
    """
    #Feature: Dashboard data
    #Scenario: The facility, zones and meters the dashboard can show
    #Complexity: Time O(m) - meters

    Args:
        panel: The panel.

    Returns:
        One dict per unit with id, level and name.

    Raises:
        ValueError: When panel is None.
    """
    require("units", "panel", panel)
    out = [{"id": "facility", "level": "facility", "name": "Facility (all zones)"}]
    for z in sorted(set(panel.zone)):
        out.append({"id": f"zone-{z}", "level": "zone", "name": f"Zone {z:02d}"})
    for label, z in sorted(zip(panel.meters, panel.zone), key=lambda p: (p[1], p[0])):
        out.append({"id": label, "level": "meter", "name": _short(label), "zone": f"Zone {z:02d}"})
    return out


@traced
def _unit_key(level: str, unit: str) -> str:
    """
    #Feature: Dashboard data
    #Scenario: The dashboard's id for a scored unit
    #Complexity: Time O(1) - constant

    Args:
        level: "facility", "zone" or "meter".
        unit: Unit name from the scoring table.

    Returns:
        "facility", "zone-<n>" or the meter label.

    Raises:
        ValueError: When an argument is None.
    """
    require("_unit_key", "level", level)
    require("_unit_key", "unit", unit)
    if level == "facility":
        return "facility"
    return f"zone-{int(unit.split()[-1])}" if level == "zone" else unit


@traced
def _num(value) -> object:
    """
    #Feature: Dashboard data
    #Scenario: JSON-safe number: missing stays missing, accuracy below zero shows as 0
    #Complexity: Time O(1) - constant

    Args:
        value: A number or None.

    Returns:
        A rounded float or None.

    Raises:
        None.
    """
    if value is None or not np.isfinite(value):
        return None
    return round(max(float(value), 0.0), 1)


@traced
def accuracy_block(levels: pd.DataFrame) -> Dict:
    """
    #Feature: Dashboard data
    #Scenario: Overall and monthly accuracy per unit and horizon
    #Complexity: Time O(n) - scored periods

    Args:
        levels: Output of all_levels.

    Returns:
        unit id -> horizon -> {overall, months, periods}.

    Raises:
        ValueError: When levels is None.
    """
    require("accuracy_block", "levels", levels)
    out: Dict = {}
    levels = levels.copy()
    levels["month"] = levels["issue_date"].dt.strftime("%Y-%m")
    for (level, unit), part in levels.groupby(["level", "unit"]):
        entry = {}
        for horizon in _HORIZONS:
            h = part[part["horizon"] == horizon]
            if h.empty:
                continue
            months = {m: _num(accuracy(g)) for m, g in h.groupby("month")}
            entry[horizon] = {"overall": _num(accuracy(h)), "months": months, "periods": int(len(h))}
        out[_unit_key(level, unit)] = entry
    return out


@traced
def calendar_month_table(levels: pd.DataFrame) -> Dict:
    """
    #Feature: Dashboard data
    #Scenario: Forecast, actual and accuracy for every whole month per unit
    #Complexity: Time O(n) - scored periods

    Args:
        levels: Output of all_levels.

    Returns:
        unit id -> list of month rows.

    Raises:
        ValueError: When levels is None.
    """
    require("calendar_month_table", "levels", levels)
    out = {}
    cm = levels[levels["horizon"] == "calendar month"]
    for (level, unit), part in cm.groupby(["level", "unit"]):
        out[_unit_key(level, unit)] = [
            {"month": d.strftime("%Y-%m"), "forecast": round(f, 1), "actual": round(a, 1), "accuracy": _num(acc)}
            for d, f, a, acc in part.sort_values("issue_date")[["issue_date", "forecast", "actual", "accuracy"]]
            .itertuples(index=False)]
    return out


@traced
def _clean(values) -> list:
    """
    #Feature: Dashboard data
    #Scenario: Round an array to one decimal, NaN becomes None
    #Complexity: Time O(n) - linear

    Args:
        values: Iterable of numbers.

    Returns:
        A JSON-safe list.

    Raises:
        ValueError: When values is None.
    """
    require("_clean", "values", values)
    return [None if np.isnan(x) else round(float(x), 1) for x in values]


@traced
def _history_arrays(panel, daily: pd.DataFrame, preds: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    #Feature: Dashboard data
    #Scenario: Actual and same-morning forecast per meter for every replayed day
    #Complexity: Time O(m*d) - meters times days

    Chart line: energy per day with outage catch-ups spread over the missing days
    (same totals as gold; gold books a whole outage on the return day).

    Args:
        panel: The panel.
        daily: The daily table.
        preds: Every-morning forecasts.

    Returns:
        (actual, forecast, dates).

    Raises:
        ValueError: When an argument is None.
    """
    require("_history_arrays", "panel", panel)
    require("_history_arrays", "preds", preds)
    day0 = preds[preds["days_ahead"] == 0]
    first, last = int(day0["target_idx"].min()), int(day0["target_idx"].max())
    hist_dates = panel.dates[first:last + 1]
    fc = np.full((len(panel.meters), last - first + 1), np.nan)
    fc[day0["meter_idx"].to_numpy(), day0["target_idx"].to_numpy() - first] = day0["pred_kwh"].to_numpy()
    complete = daily[daily["complete_day"]]
    act = (complete.pivot_table(index="meter_label", columns="date", values="spread_kwh")
           .reindex(index=panel.meters, columns=hist_dates).to_numpy())
    return act, fc, hist_dates


@traced
def series_block(panel, daily: pd.DataFrame, preds: pd.DataFrame, future: pd.DataFrame) -> Tuple[Dict, Dict]:
    """
    #Feature: Dashboard data
    #Scenario: Daily actual, forecast and future lines for the facility, zones and meters
    #Complexity: Time O(m*d) - meters times days

    Args:
        panel: The panel.
        daily: The daily table.
        preds: Every-morning forecasts.
        future: Year-end forecast.

    Returns:
        (series per unit id, {"history": [...], "future": [...]} dates).

    Raises:
        ValueError: When an argument is None.
    """
    require("series_block", "panel", panel)
    require("series_block", "future", future)
    act, fc, hist_dates = _history_arrays(panel, daily, preds)
    fut_dates = sorted(future["date"].unique())
    fut = (future.pivot_table(index="meter_label", columns="date", values="forecast_kwh")
           .reindex(panel.meters).reindex(columns=fut_dates).to_numpy())

    def pack(rows):
        a = np.nansum(act[rows], 0) if len(rows) > 1 else act[rows[0]]
        f = np.nansum(fc[rows], 0) if len(rows) > 1 else fc[rows[0]]
        a = np.where(np.isnan(act[rows]).all(0), np.nan, a)
        return {"actual": _clean(a), "forecast": _clean(f), "future": _clean(np.nansum(fut[rows], 0))}

    series = {"facility": pack(list(range(len(panel.meters))))}
    for z in sorted(set(panel.zone)):
        series[f"zone-{z}"] = pack(list(np.flatnonzero(panel.zone == z)))
    for i, label in enumerate(panel.meters):
        series[label] = pack([i])
    return series, {"history": [d.strftime("%Y-%m-%d") for d in hist_dates],
                    "future": [pd.Timestamp(d).strftime("%Y-%m-%d") for d in fut_dates]}


@traced
def _period_tables(panel, preds: pd.DataFrame, index: Dict[int, int], n_origins: int) -> Tuple[Dict, Dict]:
    """
    #Feature: Dashboard data
    #Scenario: Gold total and scorable flag of every period issued on every morning
    #Complexity: Time O(m*o) - meters times mornings

    Args:
        panel: The panel.
        preds: Every-morning forecasts.
        index: origin day index -> position among mornings.
        n_origins: Number of mornings.

    Returns:
        (actual[horizon][meter][morning], scored[horizon][meter][morning]).

    Raises:
        ValueError: When an argument is None.
    """
    require("_period_tables", "panel", panel)
    require("_period_tables", "preds", preds)
    m = len(panel.meters)
    actual, scored = {}, {}
    for h, part in meter_periods(panel, preds).groupby("horizon"):
        a = np.full((m, n_origins), np.nan)
        s = np.zeros((m, n_origins), dtype=int)
        rows = part["meter_idx"].to_numpy(), part["origin_idx"].map(index).to_numpy()
        a[rows] = part["actual"].to_numpy()
        s[rows] = part["measurable"].to_numpy().astype(int)
        actual[str(h)] = [[None if not np.isfinite(v) else round(float(v), 1) for v in row] for row in a]
        scored[str(h)] = s.tolist()
    return actual, scored


@traced
def explorer_block(panel, preds: pd.DataFrame, history_dates: List[str]) -> Dict:
    """
    #Feature: Dashboard data
    #Scenario: Everything the date explorer needs, per meter (zones and facility are sums)
    #Complexity: Time O(m*o*k) - meters, mornings, days ahead

    Args:
        panel: The panel.
        preds: Every-morning forecasts.
        history_dates: Dates of the daily chart, which the mornings must match.

    Returns:
        Dict with mornings, days, meters, measured, wf, actual and scored.

    Raises:
        ValueError: When panel or preds is None.
        Errors.HTTP.InternalServerError: When the mornings do not line up with the chart dates.
    """
    require("explorer_block", "panel", panel)
    require("explorer_block", "preds", preds)
    origins = sorted(int(t) for t in preds["origin_idx"].unique())
    origin_dates = [panel.dates[t].strftime("%Y-%m-%d") for t in origins]
    if origin_dates != list(history_dates):
        raise Errors.HTTP.InternalServerError(
            "[explorer_block] Explorer mornings do not line up with the daily chart dates. Invariant violated: "
            "same dates. Received: different lists. Fix: rerun the pipeline so both come from one run.")
    index = {t: i for i, t in enumerate(origins)}
    k, m = int(preds["days_ahead"].max()) + 1, len(panel.meters)
    wf = np.full((m, len(origins), k), np.nan)
    wf[preds["meter_idx"].to_numpy(), preds["origin_idx"].map(index).to_numpy(),
       preds["days_ahead"].to_numpy()] = preds["pred_kwh"].to_numpy()
    actual, scored = _period_tables(panel, preds, index, len(origins))
    first = origins[0]
    measured = ["".join("1" if v else "0" for v in panel.measurable[i, first:first + len(origins)])
                for i in range(m)]
    return {"mornings": origin_dates, "days": k, "meters": list(panel.meters), "measured": measured,
            "wf": [[None if not np.isfinite(v) else round(float(v), 1) for v in wf[i].ravel()] for i in range(m)],
            "actual": actual, "scored": scored}


@traced
def bills_block(bills: pd.DataFrame, breakers: pd.DataFrame) -> Dict:
    """
    #Feature: Dashboard data
    #Scenario: Per meter and 30-day cycle: predicted bill parts and the bill on actual kWh
    #Complexity: Time O(m*c) - meters times cycles

    Args:
        bills: Output of BillingService.build_bills.
        breakers: Output of BillingService.load_breakers.

    Returns:
        Dict with cycles, parts, meters and the tariff.

    Raises:
        ValueError: When an argument is None.
    """
    require("bills_block", "bills", bills)
    require("bills_block", "breakers", breakers)
    first = bills.drop_duplicates("cycle").sort_values("cycle")
    cycle_list = [{"n": int(r.cycle), "start": str(r.cycle_start), "end": str(r.cycle_end), "status": r.status,
                   "days": int(r.days_in_year), "basis": r.basis} for r in first.itertuples()]

    def nums(row, prefix):
        values = [row[f"{prefix}_{p}"] for p in _BILL_PARTS]
        return None if not np.isfinite(values[0]) else [round(float(v), 2) for v in values]

    meters = {}
    for label, part in bills.sort_values("cycle").groupby("meter_label"):
        b = breakers.set_index("meter_label").loc[label]
        meters[label] = {"breaker_a": float(b["breaker_a"]), "class": b["service_class"],
                         "fee": float(b["service_fee_sar"]),
                         "predicted": [nums(r, "predicted") for _, r in part.iterrows()],
                         "actual": [nums(r, "actual") for _, r in part.iterrows()]}
    return {"cycles": cycle_list, "parts": _BILL_PARTS, "meters": meters,
            "tariff": {"limit": TIER_LIMIT_KWH, "low": RATE_UP_TO_LIMIT, "high": RATE_ABOVE_LIMIT,
                       "vat": VAT_RATE, "classes": [[c[2], c[3]] for c in SERVICE_FEE_CLASSES]}}


@traced
def outlook_block(daily: pd.DataFrame, future: pd.DataFrame, year: int) -> Dict:
    """
    #Feature: Dashboard data
    #Scenario: Per unit: month -> actual so far plus forecast for the rest of the year
    #Complexity: Time O(n) - rows

    Args:
        daily: The daily table.
        future: Year-end forecast.
        year: The forecast year.

    Returns:
        unit id -> month -> {actual, forecast}.

    Raises:
        ValueError: When an argument is None.
    """
    require("outlook_block", "daily", daily)
    require("outlook_block", "future", future)
    require("outlook_block", "year", year)
    actual = daily[daily["complete_day"] & (daily["date"].dt.year == year)].copy()
    actual["month"] = actual["date"].dt.strftime("%Y-%m")
    fut = future.copy()
    fut["month"] = fut["date"].dt.strftime("%Y-%m")
    out: Dict = {}

    def add(key, a, f):
        months = sorted(set(a.index) | set(f.index))
        out[key] = {m: {"actual": round(float(a.get(m, 0.0)), 1), "forecast": round(float(f.get(m, 0.0)), 1)}
                    for m in months}

    add("facility", actual.groupby("month")["gold_kwh"].sum(), fut.groupby("month")["forecast_kwh"].sum())
    for z, g in actual.groupby("zone"):
        add(f"zone-{z}", g.groupby("month")["gold_kwh"].sum(), fut[fut["zone"] == z].groupby("month")["forecast_kwh"].sum())
    for m, g in actual.groupby("meter_label"):
        add(m, g.groupby("month")["gold_kwh"].sum(),
            fut[fut["meter_label"] == m].groupby("month")["forecast_kwh"].sum())
    return out
