"""The in-memory bundle one full pipeline run hands from stage to stage."""
from dataclasses import dataclass
from datetime import date
from typing import Dict

import pandas as pd


@dataclass
class PipelineResults:
    """
    #Feature: Pipeline
    #Scenario: Everything later stages (dashboard, API) need from the forecasting run
    #Complexity: Time O(1) - holds references only

    Panel, daily table, replayed forecasts, scored levels, year-end forecast and dates.
    """

    panel: object
    daily: pd.DataFrame
    preds: pd.DataFrame
    levels: pd.DataFrame
    future: pd.DataFrame
    issue: pd.Timestamp
    long_range: Dict
    forecast_end: date
    validation_start: date
    long_range_since: str

    @property
    def data_through(self) -> pd.Timestamp:
        """The last day of actual readings the forecast used (the day before the issue day)."""
        return self.issue - pd.Timedelta(days=1)
