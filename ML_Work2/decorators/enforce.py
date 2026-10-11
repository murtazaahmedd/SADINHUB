"""@enforce(permission): API-key check and permission check in one decorator."""
import functools
from typing import Any, Callable

from fastapi import Request

from errors import Errors
from utilities.auth_reforce.auth_context import AuthContext
from utilities.guard import require


def _authenticate(request: Request) -> AuthContext:
    """
    #Feature: Authorization
    #Scenario: Check the X-API-Key header when an API key is configured
    #Complexity: Time O(1) - constant

    Args:
        request: The incoming request.

    Returns:
        The auth context for this call.

    Raises:
        Errors.HTTP.Unauthorized: When a key is configured and the header is missing or wrong.
    """
    require("_authenticate", "request", request)
    key = request.app.state.settings.api_key
    if key and request.headers.get("X-API-Key") != key:
        raise Errors.HTTP.Unauthorized(
            "[enforce] Missing or invalid API key. Invariant violated: X-API-Key matches SADIN_API_KEY. "
            "Received: a different or missing header. Fix: send the X-API-Key header.")
    return AuthContext()


def enforce(permission: str) -> Callable:
    """
    #Feature: Authorization
    #Scenario: Guard an endpoint with one permission (innermost decorator)
    #Complexity: Time O(1) - constant

    After it succeeds, request.state.auth holds the AuthContext.

    Args:
        permission: A resource.action string from utilities.auth_reforce.permissions.

    Returns:
        A decorator.

    Raises:
        ValueError: When permission is None.
    """
    require("enforce", "permission", permission)

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            request: Request = kwargs["request"]
            auth = _authenticate(request)
            if permission not in auth.permissions:
                raise Errors.HTTP.Forbidden(f"[enforce] Permission {permission!r} is not granted. Invariant "
                                            f"violated: caller holds it. Received: {sorted(auth.permissions)}.")
            request.state.auth = auth
            return await func(*args, **kwargs)

        return wrapper

    return decorator
