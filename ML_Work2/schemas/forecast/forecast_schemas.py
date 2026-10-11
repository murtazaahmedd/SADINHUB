"""Schemas for saved forecasts."""
from typing import Any, Dict, List

from pydantic import BaseModel


class ForecastPageResponse(BaseModel):
    """
    #Feature: Forecast
    #Scenario: One page of the saved forecast
    #Complexity: Time O(1) - constant

    Total rows, page, limit and the rows themselves.
    """

    total: int
    page: int
    limit: int
    rows: List[Dict[str, Any]]
