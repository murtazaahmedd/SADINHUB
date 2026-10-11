"""Schemas for alerts."""
from typing import Any, Dict, List

from pydantic import BaseModel


class AlertsPageResponse(BaseModel):
    """
    #Feature: Alerts
    #Scenario: One page of consumption alerts
    #Complexity: Time O(1) - constant

    Total rows, page, limit and the rows themselves.
    """

    total: int
    page: int
    limit: int
    rows: List[Dict[str, Any]]


class DataProblemsResponse(BaseModel):
    """
    #Feature: Alerts
    #Scenario: Data-problem episodes
    #Complexity: Time O(1) - constant

    Total episodes and the rows.
    """

    total: int
    rows: List[Dict[str, Any]]


class AlertRunResponse(BaseModel):
    """
    #Feature: Alerts
    #Scenario: What an alert run produced
    #Complexity: Time O(1) - constant

    Alerts per period and the final-test agreement.
    """

    periods: List[Dict[str, Any]]
    final_test_agreement: Dict[str, int]


class IncidentsPageResponse(BaseModel):
    """
    #Feature: Alerts
    #Scenario: One page of incidents (repeated alert days grouped into one item each)
    #Complexity: Time O(1) - constant

    Total incidents, page, limit and the rows.
    """

    total: int
    page: int
    limit: int
    rows: List[Dict[str, Any]]
