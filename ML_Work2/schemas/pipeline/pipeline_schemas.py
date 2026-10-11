"""Schemas for running the pipeline."""
from typing import Dict

from pydantic import BaseModel, Field


class RunPipelineRequest(BaseModel):
    """
    #Feature: Pipeline
    #Scenario: Ask for a full run
    #Complexity: Time O(1) - constant

    Whether to re-read the warehouse and Excel exports first.
    """

    refresh_sources: bool = Field(False, description="Re-read the warehouse and the Excel exports")


class PipelineRunResponse(BaseModel):
    """
    #Feature: Pipeline
    #Scenario: What a full run produced
    #Complexity: Time O(1) - constant

    Key dates, long-range accuracy and the dashboard path.
    """

    data_through: str
    forecast_issued: str
    forecast_end: str
    long_range_accuracy: Dict[str, float]
    dashboard: str


class DashboardResponse(BaseModel):
    """
    #Feature: Pipeline
    #Scenario: Where the rebuilt dashboard is
    #Complexity: Time O(1) - constant

    Path of the dashboard page.
    """

    dashboard: str
