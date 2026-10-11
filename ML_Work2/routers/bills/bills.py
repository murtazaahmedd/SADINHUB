"""GET /bills: the saved bill forecast."""
from typing import Optional

from fastapi import APIRouter, Depends, Query, Request
from starlette.concurrency import run_in_threadpool

from decorators.enforce import enforce
from decorators.http_error_handler import http_error_handler
from main.dependencies import get_service_container
from schemas.bills.bill_schemas import BillsResponse
from services.container import ServiceContainer
from utilities.auth_reforce.permissions import bills

router = APIRouter(prefix="/bills", tags=["bills"])


@http_error_handler
@router.get("", response_model=BillsResponse)
@enforce(bills.read)
async def read_bills(request: Request, cycle: Optional[int] = Query(None, ge=1),
                     services: ServiceContainer = Depends(get_service_container)) -> BillsResponse:
    """
    #Feature: Bills
    #Scenario: The saved bill forecast, optionally one 30-day cycle
    #Complexity: Time O(n) - rows

    Args:
        request: The incoming request (carries request.state.auth).
        cycle: Optional cycle number.
        services: The service container.

    Returns:
        Total rows and the rows.

    Raises:
        Errors.Misc.DataNotReady: When the pipeline has not run yet.
    """
    return BillsResponse(**await run_in_threadpool(services.billing.read_bills, cycle))
