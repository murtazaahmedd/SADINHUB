"""FastAPI dependencies. The container is built once, in main.app, and kept on app.state."""
from fastapi import Request

from services.container import ServiceContainer


def get_service_container(request: Request) -> ServiceContainer:
    """
    #Feature: Dependency injection
    #Scenario: Hand the one service container to an endpoint
    #Complexity: Time O(1) - constant

    Args:
        request: The incoming request.

    Returns:
        The application's ServiceContainer.

    Raises:
        AttributeError: When the application was created without a container.
    """
    return request.app.state.services
