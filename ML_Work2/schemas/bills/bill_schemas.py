"""Schemas for the bill forecast."""
from typing import Any, Dict, List

from pydantic import BaseModel


class BillsResponse(BaseModel):
    """
    #Feature: Bills
    #Scenario: The saved bill forecast
    #Complexity: Time O(1) - constant

    Total rows and the rows.
    """

    total: int
    rows: List[Dict[str, Any]]
