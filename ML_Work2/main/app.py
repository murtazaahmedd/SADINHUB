"""The FastAPI application and its composition root."""
from typing import Optional

from fastapi import FastAPI

from config.settings import Settings, load_settings
from decorators.http_error_handler import register_error_handlers
from repositories.container import RepositoryContainer
from routers.accuracy.accuracy import router as accuracy_router
from routers.alerts.alerts import router as alerts_router
from routers.bills.bills import router as bills_router
from routers.forecast.forecast import router as forecast_router
from routers.health.health import router as health_router
from routers.pipeline.pipeline import router as pipeline_router
from services.container import ServiceContainer
from utilities.guard import traced


@traced
def build_service_container(settings: Settings) -> ServiceContainer:
    """
    #Feature: Composition root
    #Scenario: The one place where repositories and services are wired together
    #Complexity: Time O(1) - constant

    Args:
        settings: Folders and connection settings.

    Returns:
        A ServiceContainer (lazy: nothing connects until used).

    Raises:
        ValueError: When settings is None.
    """
    return ServiceContainer(RepositoryContainer(settings))


@traced
def create_app(settings: Optional[Settings] = None) -> FastAPI:
    """
    #Feature: Composition root
    #Scenario: Build the application, register the routers and the error mapping
    #Complexity: Time O(1) - constant

    Args:
        settings: Optional settings (read from the environment when omitted).

    Returns:
        The FastAPI application.

    Raises:
        ValueError: When SADINHUB_PORT is not an integer.
    """
    settings = settings or load_settings()
    app = FastAPI(title="SADIN energy forecasting", version="1.0")
    app.state.settings = settings
    app.state.services = build_service_container(settings)
    register_error_handlers(app)
    for router in (health_router, pipeline_router, forecast_router, accuracy_router, alerts_router, bills_router):
        app.include_router(router)
    return app


app = create_app()
