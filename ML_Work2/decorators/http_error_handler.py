"""Map application errors to HTTP responses.

With plain FastAPI the outermost decorator wraps the function *after* the router
registered it, so a wrapper there cannot change behaviour. The mapping therefore
lives in register_error_handlers(app); @http_error_handler marks each endpoint so
a test can prove every route is covered.
"""
import logging
from typing import Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from errors import AppError
from utilities.guard import require

logger = logging.getLogger(__name__)
MARKER = "__http_error_handled__"


def http_error_handler(func: Callable) -> Callable:
    """
    #Feature: Error mapping
    #Scenario: Mark an endpoint as covered by the application's HTTP error mapping
    #Complexity: Time O(1) - constant

    Args:
        func: The endpoint.

    Returns:
        The same function, marked.

    Raises:
        ValueError: When func is None.
    """
    require("http_error_handler", "func", func)
    setattr(func, MARKER, True)
    return func


async def _app_error(request: Request, exc: AppError) -> JSONResponse:
    """
    #Feature: Error mapping
    #Scenario: An AppError becomes its own status code
    #Complexity: Time O(1) - constant

    Args:
        request: The request.
        exc: The error.

    Returns:
        JSON {"detail": message} with the error's status.

    Raises:
        None.
    """
    logger.warning("%s on %s: %s", type(exc).__name__, request.url.path, exc.message)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})


async def _value_error(request: Request, exc: ValueError) -> JSONResponse:
    """
    #Feature: Error mapping
    #Scenario: A ValueError (bad input that reached a service) becomes 400
    #Complexity: Time O(1) - constant

    Args:
        request: The request.
        exc: The error.

    Returns:
        JSON {"detail": message} with status 400.

    Raises:
        None.
    """
    logger.warning("ValueError on %s: %s", request.url.path, exc)
    return JSONResponse(status_code=400, content={"detail": str(exc)})


def register_error_handlers(app: FastAPI) -> None:
    """
    #Feature: Error mapping
    #Scenario: Install the mapping on the application
    #Complexity: Time O(1) - constant

    Args:
        app: The FastAPI application.

    Raises:
        ValueError: When app is None.
    """
    require("register_error_handlers", "app", app)
    app.add_exception_handler(AppError, _app_error)
    app.add_exception_handler(ValueError, _value_error)
