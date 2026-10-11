"""POST /alerts/run, GET /alerts and GET /alerts/data-problems."""
from typing import Literal, Optional

from fastapi import APIRouter, Depends, Query, Request
from starlette.concurrency import run_in_threadpool

from decorators.enforce import enforce
from decorators.http_error_handler import http_error_handler
from main.dependencies import get_service_container
from schemas.alerts.alert_schemas import (AlertRunResponse, AlertsPageResponse, DataProblemsResponse,
                                          IncidentsPageResponse)
from services.container import ServiceContainer
from utilities.auth_reforce.permissions import alerts

router = APIRouter(prefix="/alerts", tags=["alerts"])


@http_error_handler
@router.post("/run", response_model=AlertRunResponse)
@enforce(alerts.execute)
async def run_alerts(request: Request,
                     services: ServiceContainer = Depends(get_service_container)) -> AlertRunResponse:
    """
    #Feature: Alerts
    #Scenario: Recompute the alerts from the saved forecasts and rebuild the dashboard tab
    #Complexity: Time O(n * trees) - meter-days times forest size

    Args:
        request: The incoming request (carries request.state.auth).
        services: The service container.

    Returns:
        Alerts per period and the final-test agreement.

    Raises:
        Errors.Misc.DataNotReady: When the pipeline has not run yet.
        Errors.Misc.InsufficientData: When the development period has nothing to train on.
    """
    return AlertRunResponse(**await run_in_threadpool(services.anomaly.run))


@http_error_handler
@router.get("", response_model=AlertsPageResponse)
@enforce(alerts.read)
async def read_alerts(request: Request, period: Optional[str] = Query(None),
                      detector: Optional[Literal["rule", "ml", "both"]] = Query(None),
                      meter: Optional[str] = Query(None), page: int = Query(1, ge=1, le=1_000_000),
                      limit: int = Query(50, ge=1, le=500),
                      services: ServiceContainer = Depends(get_service_container)) -> AlertsPageResponse:
    """
    #Feature: Alerts
    #Scenario: Page through the saved consumption alerts
    #Complexity: Time O(n) - alerts

    Args:
        request: The incoming request (carries request.state.auth).
        period: Optional period name.
        detector: Optional detector filter.
        meter: Optional meter label.
        page: 1-based page.
        limit: Rows per page.
        services: The service container.

    Returns:
        One page of alerts.

    Raises:
        Errors.Misc.DataNotReady: When the alerts have not been computed yet.
    """
    return AlertsPageResponse(**await run_in_threadpool(services.anomaly.read_alerts, period, detector, meter,
                                                        page, limit))


@http_error_handler
@router.get("/data-problems", response_model=DataProblemsResponse)
@enforce(alerts.read)
async def read_data_problems(request: Request,
                             services: ServiceContainer = Depends(get_service_container)) -> DataProblemsResponse:
    """
    #Feature: Alerts
    #Scenario: The saved data-problem episodes
    #Complexity: Time O(n) - episodes

    Args:
        request: The incoming request (carries request.state.auth).
        services: The service container.

    Returns:
        Total episodes and the rows.

    Raises:
        Errors.Misc.DataNotReady: When the alerts have not been computed yet.
    """
    return DataProblemsResponse(**await run_in_threadpool(services.anomaly.read_data_problems))


@http_error_handler
@router.get("/incidents", response_model=IncidentsPageResponse)
@enforce(alerts.read)
async def read_incidents(request: Request, active_only: bool = Query(False), page: int = Query(1, ge=1, le=1_000_000),
                         limit: int = Query(50, ge=1, le=500),
                         services: ServiceContainer = Depends(get_service_container)) -> IncidentsPageResponse:
    """
    #Feature: Alerts
    #Scenario: Repeated alert days grouped into incidents, biggest impact first
    #Complexity: Time O(n) - incidents

    Args:
        request: The incoming request (carries request.state.auth).
        active_only: Only incidents still running at the end of the data.
        page: 1-based page.
        limit: Rows per page.
        services: The service container.

    Returns:
        One page of incidents.

    Raises:
        Errors.Misc.DataNotReady: When the alerts have not been computed yet.
    """
    return IncidentsPageResponse(**await run_in_threadpool(services.anomaly.read_incidents, active_only, page, limit))


@http_error_handler
@router.get("/night-load", response_model=DataProblemsResponse)
@enforce(alerts.read)
async def read_night_load(request: Request,
                          services: ServiceContainer = Depends(get_service_container)) -> DataProblemsResponse:
    """
    #Feature: Alerts
    #Scenario: Nights clearly above the meter's own usual night load (equipment left running)
    #Complexity: Time O(n) - alerts

    Args:
        request: The incoming request (carries request.state.auth).
        services: The service container.

    Returns:
        Total alerts and the rows.

    Raises:
        Errors.Misc.DataNotReady: When the alerts have not been computed yet.
    """
    return DataProblemsResponse(**await run_in_threadpool(services.anomaly.read_night_alerts))
