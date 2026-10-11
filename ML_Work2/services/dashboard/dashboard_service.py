"""Build the single-file dashboard: predictions, accuracy, bills and alerts."""
import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from constants.forecasting import ACCURACY_TARGET
from interfaces.repositories.artifacts.artifact_repository import IArtifactRepository
from interfaces.services.billing.billing_service import IBillingService
from interfaces.services.dashboard.dashboard_service import IDashboardService
from utilities.dashboard_payload import (accuracy_block, bills_block, calendar_month_table, explorer_block,
                                         outlook_block, series_block, units)
from utilities.guard import require, traced

ASSETS = Path(__file__).resolve().parent / "assets"


class DashboardService(IDashboardService):
    """
    #Feature: Dashboard
    #Scenario: Assemble data blocks into the offline HTML page
    #Complexity: Time O(n) - page data

    Uses the billing service for bills and the store for the page and the alerts.
    """

    @traced
    def __init__(self, artifacts: IArtifactRepository, billing: IBillingService) -> None:
        """
        #Feature: Dashboard
        #Scenario: Constructor injection of the store and the billing service
        #Complexity: Time O(1) - constant

        Args:
            artifacts: Saved-results store.
            billing: Bill forecast service.

        Raises:
            ValueError: When an argument is None.
        """
        require("DashboardService.__init__", "artifacts", artifacts)
        require("DashboardService.__init__", "billing", billing)
        self._artifacts = artifacts
        self._billing = billing

    @traced
    def _alerts_block(self) -> Optional[Dict]:
        """
        #Feature: Dashboard
        #Scenario: The Alerts tab's data, marked stale when the forecasts changed since it was computed
        #Complexity: Time O(n) - file size

        Returns:
            The alerts block, or None when the anomaly step has not run.

        Raises:
            None.
        """
        if not self._artifacts.exists("alerts_dashboard"):
            return None
        block = self._artifacts.load_json("alerts_dashboard")
        recorded = block["meta"].get("inputs", {})
        current = {name: self._artifacts.fingerprint(name) for name in recorded if self._artifacts.exists(name)}
        block["meta"]["stale"] = recorded != current
        return block

    @traced
    def _data(self, r) -> Dict:
        """
        #Feature: Dashboard
        #Scenario: Every block of data the page needs
        #Complexity: Time O(n) - page data

        Args:
            r: A PipelineResults bundle.

        Returns:
            The JSON-ready data dict.

        Raises:
            ValueError: When r is None.
        """
        require("DashboardService._data", "r", r)
        series, dates = series_block(r.panel, r.daily, r.preds, r.future)
        issue = r.issue
        meta = {"issued": f"{issue:%d %B %Y}", "data_through": f"{issue - pd.Timedelta(days=1):%d %B %Y}",
                "forecast_end": f"{r.forecast_end:%d %B %Y}", "validation_from": f"{r.validation_start:%d %B %Y}",
                "target": ACCURACY_TARGET, "issue_date": f"{issue:%Y-%m-%d}", "year": r.forecast_end.year,
                "long_range_since": r.long_range_since}
        breakers = self._billing.load_breakers()
        bills = self._billing.build_bills(r.panel, r.daily, r.preds, r.future, issue, breakers, r.forecast_end)
        return {"long_range": r.long_range, "meta": meta, "units": units(r.panel),
                "accuracy": accuracy_block(r.levels), "calendar_months": calendar_month_table(r.levels),
                "series": series, "dates": dates, "outlook": outlook_block(r.daily, r.future, r.forecast_end.year),
                "explorer": explorer_block(r.panel, r.preds, dates["history"]),
                "bills": bills_block(bills, breakers), "alerts": self._alerts_block()}

    @traced
    def build(self, results) -> Path:
        """
        #Feature: Dashboard
        #Scenario: Assemble the page from a pipeline run
        #Complexity: Time O(n) - page data

        Args:
            results: A PipelineResults bundle.

        Returns:
            Path of the written HTML file.

        Raises:
            ValueError: When results is None.
        """
        require("DashboardService.build", "results", results)
        data = self._data(results)
        year = results.forecast_end.year
        html = (ASSETS / "dashboard_template.html").read_text(encoding="utf-8")
        html = html.replace("/*__CHART_JS__*/", (ASSETS / "chart.umd.min.js").read_text(encoding="utf-8"))
        html = html.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
        html = html.replace("__NEXT_YEAR__", str(year + 1)).replace("__YEAR__", str(year))
        return self._artifacts.save_text("dashboard", html)
