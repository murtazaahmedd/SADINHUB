"""One place that calls an api.* function and returns a DataFrame."""
from typing import Optional

import pandas as pd
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from errors import Errors
from interfaces.database import ISessionFactory
from utilities.guard import require, traced


@traced
def call_udf(sessions: ISessionFactory, statement: str, params: Optional[dict] = None) -> pd.DataFrame:
    """
    #Feature: Warehouse access
    #Scenario: Run one api.* function through a read-only session
    #Complexity: Time O(n) - rows returned

    Args:
        sessions: Source of read-only connections.
        statement: SQL such as "SELECT * FROM api.get_energy_meters()".
        params: Bind parameters.

    Returns:
        The rows as a DataFrame.

    Raises:
        ValueError: When sessions or statement is None.
        Errors.HTTP.InternalServerError: When the warehouse call fails.
    """
    require("call_udf", "sessions", sessions)
    require("call_udf", "statement", statement)
    try:
        with sessions.session() as conn:
            result = conn.execute(text(statement), params or {})
            return pd.DataFrame(result.fetchall(), columns=list(result.keys()))
    except SQLAlchemyError as exc:
        raise Errors.HTTP.InternalServerError(
            f"[call_udf] Warehouse call failed: {statement}. Invariant violated: the database answers the UDF. "
            f"Received: {type(exc).__name__}: {exc}. Fix: check the connection settings and that "
            f"`python -m database.udf_load` has been run.") from exc
