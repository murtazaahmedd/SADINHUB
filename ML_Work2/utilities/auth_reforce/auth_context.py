"""The authorization context placed on request.state.auth after @enforce succeeds."""
from typing import FrozenSet

from pydantic import BaseModel, ConfigDict

from utilities.auth_reforce.permissions import ALL_PERMISSIONS


class AuthContext(BaseModel):
    """
    #Feature: Authorization
    #Scenario: Who is calling and what they may do
    #Complexity: Time O(1) - constant

    This project has no user directory: a valid API key (or an open local server)
    grants every permission to a single "local" user.
    """

    model_config = ConfigDict(frozen=True)

    user_id: str = "local"
    role: str = "admin"
    permissions: FrozenSet[str] = ALL_PERMISSIONS
