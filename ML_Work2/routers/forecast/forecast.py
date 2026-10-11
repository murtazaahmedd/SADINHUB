"""GET /forecast/{level}: the saved year-end forecast."""
from typing import Literal, Optional

from fastapi import APIRouter, Depends, Query, Request
from starlette.concurrency import run_in_threadpool

from decorators.enforce import enforce
from decorators.http_error_handler import http_error_handler
from main.dependencies import get_service_container
from schemas.forecast.forecast_schemas import ForecastPageResponse
from services.container import ServiceContainer
from utilities.auth_reforce.permissions import forecast

router = APIRouter(prefix="/forecast", tags=["forecast"])


@http_error_handler
@router.get("/{level}", response_model=ForecastPageResponse)
@enforce(forecast.read)
async def read_forecast(level: Literal["meters", "zones", "facility"], request: Request,
                        start: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
                        end: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
                        page: int = Query(1, ge=1, le=1_000_000), limit: int = Query(100, ge=1, le=1000),
                        services: ServiceContainer = Depends(get_service_container)) -> ForecastPageResponse:
    """
    #Feature: Forecast
    #Scenario: Page through the saved daily forecast of meters, zones or the facility
    #Complexity: Time O(n) - rows

    Args:
        level: "meters", "zones" or "facility".
        request: The incoming request (carries request.state.auth).
        start: Optional first date.
        end: Optional last date.
        page: 1-based page.
        limit: Rows per page.
        services: The service container.

    Returns:
        One page of forecast rows.

    Raises:
        Errors.Misc.DataNotReady: When the pipeline has not run yet.
    """
    return ForecastPageResponse(**await run_in_threadpool(services.forecast.read_forecast, level, start, end,
                                                          page, limit))
