"""Error taxonomy. Raise these, never a raw HTTPException."""


class AppError(Exception):
    """Base class: every application error carries the HTTP status it maps to."""

    status_code: int = 500

    def __init__(self, message: str = "") -> None:
        super().__init__(message)
        self.message = message


class Errors:
    """Namespace for every error class used in the project."""

    class HTTP:
        """Errors that map one-to-one onto an HTTP status."""

        class BadRequest(AppError):
            status_code = 400

        class Unauthorized(AppError):
            status_code = 401

        class Forbidden(AppError):
            status_code = 403

        class NotFound(AppError):
            status_code = 404

        class InternalServerError(AppError):
            status_code = 500

    class Database:
        """Database failures."""

        class IntegrityError(AppError):
            status_code = 409

    class Misc:
        """Domain-specific business-rule violations."""

        class DataNotReady(AppError):
            """A required saved result does not exist yet; run the pipeline first."""
            status_code = 409

        class HorizonExceeded(AppError):
            """A forecast needs more days ahead than the model was trained for."""
            status_code = 422

        class MeterMappingError(AppError):
            """An export or reading cannot be tied to exactly one meter."""
            status_code = 422

        class InsufficientData(AppError):
            """Not enough history to compute the requested result."""
            status_code = 422
