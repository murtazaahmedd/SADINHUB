"""POST /pipeline/run and POST /pipeline/dashboard."""
from fastapi import APIRouter, Depends, Request
from starlette.concurrency import run_in_threadpool

from decorators.enforce import enforce
from decorators.http_error_handler import http_error_handler
from main.dependencies import get_service_container
from schemas.pipeline.pipeline_schemas import DashboardResponse, PipelineRunResponse, RunPipelineRequest
from services.container import ServiceContainer
from utilities.auth_reforce.permissions import pipeline

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


@http_error_handler
@router.post("/run", response_model=PipelineRunResponse)
@enforce(pipeline.execute)
async def run_pipeline(body: RunPipelineRequest, request: Request,
                       services: ServiceContainer = Depends(get_service_container)) -> PipelineRunResponse:
    """
    #Feature: Pipeline
    #Scenario: Run data -> validation -> forecast -> dashboard
    #Complexity: Time O(months * fit) - one model per month

    Args:
        body: Whether to refresh the sources.
        request: The incoming request (carries request.state.auth).
        services: The service container.

    Returns:
        Key dates, long-range accuracy and the dashboard path.

    Raises:
        Errors.Misc.MeterMappingError: When an Excel export cannot be tied to one meter.
        Errors.Misc.HorizonExceeded: When the forecast needs more days than the model was trained for.
    """
    results = await run_in_threadpool(services.pipeline.run, body.refresh_sources)
    path = await run_in_threadpool(services.pipeline.build_dashboard, results)
    return PipelineRunResponse(
        data_through=f"{results.data_through:%Y-%m-%d}",
        forecast_issued=f"{results.issue:%Y-%m-%d}", forecast_end=f"{results.forecast_end:%Y-%m-%d}",
        long_range_accuracy=results.long_range, dashboard=str(path))


@http_error_handler
@router.post("/dashboard", response_model=DashboardResponse)
@enforce(pipeline.execute)
async def rebuild_dashboard(request: Request,
                            services: ServiceContainer = Depends(get_service_container)) -> DashboardResponse:
    """
    #Feature: Pipeline
    #Scenario: Rebuild the dashboard page from the last saved run (no retraining)
    #Complexity: Time O(n) - page data

    Args:
        request: The incoming request (carries request.state.auth).
        services: The service container.

    Returns:
        The dashboard path.

    Raises:
        Errors.Misc.DataNotReady: When the pipeline has not run yet.
    """
    return DashboardResponse(dashboard=str(await run_in_threadpool(services.pipeline.build_dashboard)))
