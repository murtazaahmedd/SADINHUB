"""Read-only warehouse connections. The pipeline can never write to the database."""
import re
from contextlib import contextmanager
from typing import Iterator, Optional

from sqlalchemy import Connection, Engine, create_engine, event, text
from sqlalchemy.exc import InvalidRequestError

from config.settings import Settings
from errors import Errors
from interfaces.database import ISessionFactory
from utilities.guard import require, traced

READ_ONLY_OPTIONS = "-c default_transaction_read_only=on -c statement_timeout=900000"
# The only statement shape a pipeline session may run: a call to one api.* function.
ALLOWED_STATEMENT = re.compile(r"^\s*SELECT\s+\*\s+FROM\s+api\.[a-z_][a-z0-9_]*\s*\(.*\)\s*$", re.IGNORECASE | re.DOTALL)


@traced
def _only_udf_calls(conn, cursor, statement, parameters, context, executemany) -> None:
    """
    #Feature: Database access
    #Scenario: Refuse anything but a call to an api.* function, so a session cannot switch off read-only mode
    #Complexity: Time O(len(statement)) - linear

    Read-only is a session default that a session could change itself (SET default_transaction_read_only = off).
    This guard makes that impossible through this factory: no SET, DDL or DML can be issued at all.

    Args:
        conn: The connection (unused).
        cursor: The DBAPI cursor (unused).
        statement: The SQL about to run.
        parameters: Bind parameters (unused).
        context: Execution context (unused).
        executemany: Whether this is a batch (unused).

    Raises:
        InvalidRequestError: When the statement is not a call to an api.* function.
    """
    if not ALLOWED_STATEMENT.match(statement) or ";" in statement:
        raise InvalidRequestError(
            f"[ReadOnlySessionFactory] Refused statement {statement[:80]!r}. Invariant violated: pipeline sessions "
            f"only run 'SELECT * FROM api.<function>(...)'. Received: another statement. "
            f"Fix: add an api.* function and call that.")


@traced
def _verify_read_only(conn: Connection) -> None:
    """
    #Feature: Database access
    #Scenario: Prove the session cannot write before any query runs
    #Complexity: Time O(1) - one round trip

    Args:
        conn: An open connection.

    Raises:
        ValueError: When conn is None.
        Errors.HTTP.InternalServerError: When the session is not read-only.
    """
    require("_verify_read_only", "conn", conn)
    state = conn.execute(text("SHOW transaction_read_only")).scalar()
    if state != "on":
        raise Errors.HTTP.InternalServerError(
            "[_verify_read_only] Database session is not read-only. Invariant violated: "
            f"transaction_read_only must be 'on'. Received: {state!r}. Fix: check READ_ONLY_OPTIONS.")


class ReadOnlySessionFactory(ISessionFactory):
    """
    #Feature: Database access
    #Scenario: Open read-only connections lazily, so API reads work without a database
    #Complexity: Time O(1) - constant

    Creates the engine on first use and verifies every connection is read-only.
    """

    @traced
    def __init__(self, settings: Settings) -> None:
        """
        #Feature: Database access
        #Scenario: Remember the settings; connect later
        #Complexity: Time O(1) - constant

        Args:
            settings: Connection settings.

        Raises:
            ValueError: When settings is None.
        """
        require("ReadOnlySessionFactory.__init__", "settings", settings)
        self._settings = settings
        self._engine: Optional[Engine] = None

    @traced
    def _get_engine(self) -> Engine:
        """
        #Feature: Database access
        #Scenario: Create the engine the first time a query needs it
        #Complexity: Time O(1) - constant

        Returns:
            The shared SQLAlchemy engine.

        Raises:
            ValueError: When the password is missing.
        """
        if self._engine is None:
            self._engine = create_engine(self._settings.database_url,
                                         connect_args={"options": READ_ONLY_OPTIONS})
        return self._engine

    @contextmanager
    def session(self) -> Iterator[Connection]:
        """
        #Feature: Database access
        #Scenario: A verified read-only connection for one unit of work
        #Complexity: Time O(1) - constant

        Yields:
            A SQLAlchemy Connection.

        Raises:
            Errors.HTTP.InternalServerError: When the session is not read-only.
        """
        with self._get_engine().connect() as conn:
            _verify_read_only(conn)
            event.listen(conn, "before_cursor_execute", _only_udf_calls)
            yield conn
