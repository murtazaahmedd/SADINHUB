"""Bill forecast: kWh -> SAR for each meter and each 30-day billing cycle.

Which kWh is used:
  * completed cycles - the forecast made on the cycle's first morning (and,
    alongside, the bill on the actual metered kWh);
  * the current cycle - actual kWh so far plus the latest forecast for the rest;
  * future cycles     - the latest forecast.
"""
from datetime import date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from constants.tariff import CYCLE_DAYS
from errors import Errors
from interfaces.repositories.artifacts.artifact_repository import IArtifactRepository
from interfaces.repositories.readings.meter_repository import IMeterRepository
from interfaces.services.billing.billing_service import IBillingService
from utilities.billing_rules import bill, cycles, service_class
from utilities.guard import require, traced


@traced
def _cycle_status(cycle: Dict, last_complete: pd.Timestamp) -> str:
    """
    #Feature: Bill forecast
    #Scenario: Is a cycle finished, running, upcoming or running past the forecast?
    #Complexity: Time O(1) - constant

    Args:
        cycle: A cycle dict from billing_rules.cycles.
        last_complete: Last complete data day.

    Returns:
        "completed", "in progress", "upcoming" or "runs into next year".

    Raises:
        ValueError: When an argument is None.
    """
    require("_cycle_status", "cycle", cycle)
    require("_cycle_status", "last_complete", last_complete)
    start, end = pd.Timestamp(cycle["start"]), pd.Timestamp(cycle["end"])
    if not cycle["full"]:
        return "runs into next year"
    if end <= last_complete:
        return "completed"
    return "in progress" if start <= last_complete else "upcoming"


class BillingService(IBillingService):
    """
    #Feature: Bill forecast
    #Scenario: Build and serve the bill forecast
    #Complexity: Time O(m*c) - meters times cycles

    Uses the meter repository for breaker ratings and the store for results.
    """

    @traced
    def __init__(self, meters: IMeterRepository, artifacts: IArtifactRepository) -> None:
        """
        #Feature: Bill forecast
        #Scenario: Constructor injection of the meter source and the store
        #Complexity: Time O(1) - constant

        Args:
            meters: Meter reference data.
            artifacts: Saved-results store.

        Raises:
            ValueError: When an argument is None.
        """
        require("BillingService.__init__", "meters", meters)
        require("BillingService.__init__", "artifacts", artifacts)
        self._meters = meters
        self._artifacts = artifacts

    @traced
    def load_breakers(self, refresh: bool = False) -> pd.DataFrame:
        """
        #Feature: Bill forecast
        #Scenario: Breaker rating, service class and monthly fee per meter, matched on meter key
        #Complexity: Time O(m) - meters

        Args:
            refresh: Re-read from the warehouse instead of the saved copy.

        Returns:
            Columns meter_key, meter_label, breaker_a, service_class, service_fee_sar.

        Raises:
            Errors.Misc.InsufficientData: When a meter has no breaker rating.
        """
        if self._artifacts.exists("meter_breakers") and not refresh:
            return self._artifacts.load_frame("meter_breakers")
        out = self._meters.get_breakers().rename(columns={"cb_max_rating_a": "breaker_a"})
        missing = out[out["breaker_a"].isna()]
        if not missing.empty:
            raise Errors.Misc.InsufficientData(
                f"[BillingService.load_breakers] No breaker rating for: {', '.join(missing['meter_label'])}. "
                f"Invariant violated: every meter has a rating. Received: NULL. "
                f"Fix: load silver.dim_meter_cb_rating for these meters.")
        out["breaker_a"] = out["breaker_a"].astype(float)
        out[["service_class", "service_fee_sar"]] = [service_class(a) for a in out["breaker_a"]]
        self._artifacts.save_frame("meter_breakers", out)
        return out

    @traced
    def _predicted_kwh(self, label: str, status: str, days: pd.DatetimeIndex, last_complete: pd.Timestamp,
                       gold: pd.DataFrame, fut: pd.DataFrame, made_on: Optional[pd.DataFrame], i: int) -> float:
        """
        #Feature: Bill forecast
        #Scenario: The kWh a cycle's predicted bill is based on
        #Complexity: Time O(d) - days in the cycle

        Args:
            label: Meter label.
            status: Cycle status.
            days: Days of the cycle inside the forecast.
            last_complete: Last complete data day.
            gold: Gold kWh, meters x dates.
            fut: Year-end forecast, meters x dates.
            made_on: Forecasts issued on the cycle's first morning (completed cycles only).
            i: Meter index.

        Returns:
            Predicted kWh for the cycle.

        Raises:
            ValueError: When label or status is None.
        """
        require("BillingService._predicted_kwh", "label", label)
        require("BillingService._predicted_kwh", "status", status)
        if status == "completed":
            return float(made_on.loc[made_on["meter_idx"] == i, "pred_kwh"].sum())
        past = [d for d in days if d <= last_complete]
        ahead = [d for d in days if d > last_complete]
        predicted = float(gold.loc[label, past].sum()) if past else 0.0
        return predicted + (float(fut.loc[label, ahead].sum()) if ahead else 0.0)

    @traced
    def _basis_text(self, status: str, start: pd.Timestamp, issue: pd.Timestamp, last_complete: pd.Timestamp,
                    days: pd.DatetimeIndex) -> str:
        """
        #Feature: Bill forecast
        #Scenario: A plain-language note on what the predicted bill is built from
        #Complexity: Time O(d) - days in the cycle

        Args:
            status: Cycle status.
            start: First day of the cycle.
            issue: Issue day of the year-end forecast.
            last_complete: Last complete data day.
            days: Days of the cycle inside the forecast.

        Returns:
            The basis text.

        Raises:
            ValueError: When an argument is None.
        """
        require("BillingService._basis_text", "status", status)
        require("BillingService._basis_text", "start", start)
        if status == "completed":
            return f"forecast made on {start:%d %b %Y}"
        if any(d <= last_complete for d in days):
            return f"actual to {last_complete:%d %b} + forecast made {issue:%d %b %Y}"
        return f"forecast made {issue:%d %b %Y}"

    @traced
    def build_bills(self, panel, daily: pd.DataFrame, preds: pd.DataFrame, future: pd.DataFrame,
                    issue: pd.Timestamp, breakers: pd.DataFrame, forecast_end) -> pd.DataFrame:
        """
        #Feature: Bill forecast
        #Scenario: One row per meter per 30-day cycle with predicted (and actual) bill
        #Complexity: Time O(m*c) - meters times cycles

        Args:
            panel: The panel.
            daily: The daily table.
            preds: Every-morning forecasts.
            future: Year-end forecast.
            issue: Issue day of the year-end forecast.
            breakers: Output of load_breakers.
            forecast_end: Last forecast day.

        Returns:
            The bill table (also saved).

        Raises:
            Errors.Misc.InsufficientData: When a meter has no breaker rating.
        """
        require("BillingService.build_bills", "panel", panel)
        require("BillingService.build_bills", "forecast_end", forecast_end)
        last_complete = daily.loc[daily["complete_day"], "date"].max()
        fee = breakers.set_index("meter_label")
        missing = sorted(set(panel.meters) - set(fee.index))
        if missing:
            raise Errors.Misc.InsufficientData(f"[BillingService.build_bills] Meters without a breaker rating: "
                                               f"{missing}. Invariant violated: every meter has a rating. "
                                               f"Fix: load the breaker table.")
        gold = (daily[daily["complete_day"]].pivot_table(index="meter_label", columns="date", values="gold_kwh")
                .reindex(index=panel.meters))
        fut = future.pivot_table(index="meter_label", columns="date", values="forecast_kwh").reindex(panel.meters)
        morning = preds[preds["days_ahead"] < CYCLE_DAYS]
        year_start = date(pd.Timestamp(forecast_end).year, 1, 1)
        rows: List[dict] = []
        for c in cycles(year_start, forecast_end):
            rows += self._cycle_rows(c, panel, fee, gold, fut, morning, last_complete, issue, forecast_end)
        out = pd.DataFrame(rows)
        num = out.select_dtypes("number").columns.difference(["cycle", "zone", "days_in_year", "breaker_a"])
        out[num] = out[num].round(2)
        self._artifacts.save_frame("bill_forecast", out)
        return out

    @traced
    def _cycle_rows(self, c: Dict, panel, fee: pd.DataFrame, gold: pd.DataFrame, fut: pd.DataFrame,
                    morning: pd.DataFrame, last_complete: pd.Timestamp, issue: pd.Timestamp,
                    forecast_end) -> List[dict]:
        """
        #Feature: Bill forecast
        #Scenario: Bill rows of every meter for one cycle
        #Complexity: Time O(m*d) - meters times days

        Args:
            c: Cycle dict.
            panel: The panel.
            fee: Breaker table indexed by meter label.
            gold: Gold kWh, meters x dates.
            fut: Year-end forecast, meters x dates.
            morning: Every-morning forecasts for the first 30 days ahead.
            last_complete: Last complete data day.
            issue: Issue day of the year-end forecast.
            forecast_end: Last forecast day.

        Returns:
            One dict per meter.

        Raises:
            ValueError: When c is None.
        """
        require("BillingService._cycle_rows", "c", c)
        start, end = pd.Timestamp(c["start"]), pd.Timestamp(c["end"])
        days = pd.date_range(start, min(end, pd.Timestamp(forecast_end)))
        status = _cycle_status(c, last_complete)
        made_on = morning[morning["origin_idx"] == panel.index_of(start)] if status == "completed" else None
        basis = self._basis_text(status, start, issue, last_complete, days)
        rows = []
        for i, label in enumerate(panel.meters):
            predicted = self._predicted_kwh(label, status, days, last_complete, gold, fut, made_on, i)
            actual = float(gold.loc[label, list(days)].sum()) if status == "completed" else np.nan
            p = bill(predicted, fee.loc[label, "service_fee_sar"])
            a = bill(actual, fee.loc[label, "service_fee_sar"]) if np.isfinite(actual) else {}
            row = {"cycle": c["cycle"], "cycle_start": start.date(), "cycle_end": end.date(), "status": status,
                   "days_in_year": len(days), "meter_label": label, "zone": int(panel.zone[i]),
                   "breaker_a": fee.loc[label, "breaker_a"], "service_class": fee.loc[label, "service_class"],
                   "basis": basis}
            row.update({f"predicted_{k}": v for k, v in p.items()})
            row.update({f"actual_{k}": a.get(k, np.nan) for k in p})
            rows.append(row)
        return rows

    @traced
    def read_bills(self, cycle: Optional[int]) -> Dict:
        """
        #Feature: Bill forecast
        #Scenario: The saved bill forecast, optionally one cycle
        #Complexity: Time O(n) - rows

        Args:
            cycle: Optional cycle number.

        Returns:
            Dict with total and rows.

        Raises:
            Errors.Misc.DataNotReady: When the pipeline has not run yet.
        """
        frame = self._artifacts.load_frame("bill_forecast")
        if cycle is not None:
            frame = frame[frame["cycle"] == cycle]
        return {"total": int(len(frame)), "rows": frame.replace({np.nan: None}).to_dict("records")}
