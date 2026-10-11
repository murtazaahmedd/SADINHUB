"""Error-handling and validation helpers shared by every layer."""
import functools
import logging
from typing import Any, Callable, Optional, Tuple, Type, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def require(owner: str, name: str, value: Any, expected: Optional[Tuple[Type, ...] | Type] = None) -> Any:
    """
    #Feature: Input validation
    #Scenario: A required argument is missing or has the wrong type
    #Complexity: Time O(1) - constant

    Raise ValueError when ``value`` is None, then TypeError when it is not an ``expected`` type.

    Args:
        owner: "Class.method" used in the message.
        name: Argument name.
        value: Argument value.
        expected: Optional type or tuple of types.

    Returns:
        The validated value.

    Raises:
        ValueError: When the value is None.
        TypeError: When the value is not an expected type.
    """
    if value is None:
        raise ValueError(f"[{owner}] '{name}' is required and must not be None. "
                         f"Invariant violated: all required inputs present. Received: None. "
                         f"Fix: pass a value for '{name}'.")
    if expected is not None and not isinstance(value, expected):
        raise TypeError(f"[{owner}] '{name}' has the wrong type. Invariant violated: must be {expected}. "
                        f"Received: {type(value).__name__}. Fix: convert '{name}' before calling.")
    return value


def traced(func: F) -> F:
    """
    #Feature: Error handling and observability
    #Scenario: Any function fails; the failure is logged with its context and re-raised unchanged
    #Complexity: Time O(1) - constant overhead per call

    Wrap ``func`` in a try/except that logs the failing function and re-raises the same exception.

    Args:
        func: The function to wrap.

    Returns:
        The wrapped function.

    Raises:
        Exception: Whatever ``func`` raised, unchanged.
    """
    logger = logging.getLogger(func.__module__)

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            logger.error("%s failed: %s: %s", func.__qualname__, type(exc).__name__, exc)
            raise

    return wrapper  # type: ignore[return-value]
