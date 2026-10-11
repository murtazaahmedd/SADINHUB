"""Schemas for saved accuracy tables."""
from typing import Any, Dict, List

from pydantic import BaseModel


class AccuracySummaryResponse(BaseModel):
    """
    #Feature: Accuracy
    #Scenario: The accuracy summary per level and horizon
    #Complexity: Time O(1) - constant

    One dict per level and horizon.
    """

    rows: List[Dict[str, Any]]


class MonthMatrixResponse(BaseModel):
    """
    #Feature: Accuracy
    #Scenario: Unit x month accuracy table
    #Complexity: Time O(1) - constant

    Month columns and one row per unit.
    """

    columns: List[str]
    rows: List[Dict[str, Any]]
