"""GET /accuracy/summary and GET /accuracy/by-month."""
from typing import Literal

from fastapi import APIRouter, Depends, Query, Request
from starlette.concurrency import run_in_threadpool

from decorators.enforce import enforce
from decorators.http_error_handler import http_error_handler
from main.dependencies import get_service_container
from schemas.accuracy.accuracy_schemas import AccuracySummaryResponse, MonthMatrixResponse
from services.container import ServiceContainer
from utilities.auth_reforce.permissions import accuracy

router = APIRouter(prefix="/accuracy", tags=["accuracy"])


@http_error_handler
@router.get("/summary", response_model=AccuracySummaryResponse)
@enforce(accuracy.read)
async def accuracy_summary(request: Request,
                           services: ServiceContainer = Depends(get_service_container)) -> AccuracySummaryResponse:
    """
    #Feature: Accuracy
    #Scenario: The saved accuracy summary per level and horizon
    #Complexity: Time O(1) - small file

    Args:
        request: The incoming request (carries request.state.auth).
        services: The service container.

    Returns:
        The summary rows.

    Raises:
        Errors.Misc.DataNotReady: When the pipeline has not run yet.
    """
    return AccuracySummaryResponse(rows=await run_in_threadpool(services.accuracy.read_summary))


@http_error_handler
@router.get("/by-month", response_model=MonthMatrixResponse)
@enforce(accuracy.read)
async def accuracy_by_month(request: Request, level: Literal["meter", "zone", "facility"] = Query("facility"),
                            horizon: Literal["1 day", "7 days", "30 days", "calendar month"] = Query("30 days"),
                            services: ServiceContainer = Depends(get_service_container)) -> MonthMatrixResponse:
    """
    #Feature: Accuracy
    #Scenario: The saved unit x month accuracy table
    #Complexity: Time O(n) - table size

    Args:
        request: The incoming request (carries request.state.auth).
        level: "meter", "zone" or "facility".
        horizon: Horizon label.
        services: The service container.

    Returns:
        Month columns and one row per unit.

    Raises:
        Errors.Misc.DataNotReady: When the pipeline has not run yet.
    """
    return MonthMatrixResponse(**await run_in_threadpool(services.accuracy.read_month_matrix, level, horizon))
