"""GET /health."""
from fastapi import APIRouter, Depends, Request
from starlette.concurrency import run_in_threadpool

from decorators.http_error_handler import http_error_handler
from main.dependencies import get_service_container
from schemas.health.health_schemas import HealthResponse
from services.container import ServiceContainer

router = APIRouter(tags=["health"])


@http_error_handler
@router.get("/health", response_model=HealthResponse)
async def health(request: Request, services: ServiceContainer = Depends(get_service_container)) -> HealthResponse:
    """
    #Feature: Health
    #Scenario: Is the service up and which results exist yet?
    #Complexity: Time O(1) - constant

    Args:
        request: The incoming request.
        services: The service container.

    Returns:
        Status and which saved results exist.

    Raises:
        None.
    """
    return HealthResponse(status="ok", results_available=await run_in_threadpool(services.pipeline.status))
