"""Schemas for the health endpoint."""
from typing import Dict

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """
    #Feature: Health
    #Scenario: Is the service up and which results exist yet?
    #Complexity: Time O(1) - constant

    Service status and which saved results are available.
    """

    status: str
    results_available: Dict[str, bool]
