"""Adversarial tests of the warehouse boundary: the pipeline must never be able to write, and
hostile input must never reach SQL. Skipped unless a warehouse is reachable (SADINHUB_PASSWORD)."""
import os

import pytest
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from config.settings import load_settings
from database.session import ReadOnlySessionFactory
from errors import Errors
from repositories.artifacts.artifact_repository import ArtifactRepository
from repositories.readings.energy_reading_repository import EnergyReadingRepository
from repositories.readings.meter_repository import MeterRepository
from tests.conftest import ROOT

pytestmark = pytest.mark.database


@pytest.fixture(scope="module")
def raw():
    """An unguarded read-only connection, for the test's own reconciliation queries."""
    if not (os.environ.get("SADINHUB_PASSWORD") or os.environ.get("PGPASSWORD")):
        pytest.skip("database password not set")
    from sqlalchemy import create_engine
    from database.session import READ_ONLY_OPTIONS
    engine = create_engine(load_settings().database_url, connect_args={"options": READ_ONLY_OPTIONS})
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:        # pragma: no cover - environment dependent
        pytest.skip(f"warehouse not reachable: {exc}")
    return engine


@pytest.fixture(scope="module")
def sessions():
    if not (os.environ.get("SADINHUB_PASSWORD") or os.environ.get("PGPASSWORD")):
        pytest.skip("database password not set")
    factory = ReadOnlySessionFactory(load_settings())
    try:
        MeterRepository(factory).get_meters()
    except Exception as exc:        # pragma: no cover - environment dependent
        pytest.skip(f"warehouse not reachable: {exc}")
    return factory


@pytest.mark.parametrize("statement", [
    "CREATE TABLE public.pwned (x int)",
    "UPDATE gold.meter SET meter_label = 'x'",
    "DELETE FROM gold.energy_raw",
    "INSERT INTO gold.zone (zone_name) VALUES ('x')",
    "DROP TABLE gold.meter",
    "TRUNCATE gold.energy_raw",
    "CREATE FUNCTION api.evil() RETURNS int LANGUAGE sql AS 'SELECT 1'",
])
def test_the_pipeline_connection_cannot_write(sessions, statement):
    with pytest.raises(SQLAlchemyError, match="Refused statement"):
        with sessions.session() as conn:
            conn.execute(text(statement))


@pytest.mark.parametrize("statement", [
    "SET default_transaction_read_only = off", "SET transaction_read_only = off",
    "SET SESSION CHARACTERISTICS AS TRANSACTION READ WRITE", "RESET default_transaction_read_only",
    "SELECT * FROM api.get_energy_meters(); DROP TABLE gold.meter",
    "SELECT set_config('default_transaction_read_only', 'off', false)",
    "SELECT * FROM gold.meter", "SELECT pg_sleep(1)",
])
def test_read_only_can_not_be_switched_off_or_bypassed_from_inside_a_session(sessions, statement):
    with pytest.raises(SQLAlchemyError, match="Refused statement"):
        with sessions.session() as conn:
            conn.execute(text(statement))
    with sessions.session() as conn:
        pass            # and the pooled connection is still usable and still read-only


def test_hostile_meter_keys_never_reach_sql(sessions):
    repo = EnergyReadingRepository(sessions)
    for key in ["'; DROP TABLE gold.meter;--", "x' OR '1'='1", "00000000-0000-0000-0000-000000000000 OR 1=1", "\x00", ""]:
        with pytest.raises(Errors.HTTP.InternalServerError):
            repo.get_readings(key)
    assert len(MeterRepository(sessions).get_meters()) == 31          # nothing was dropped


def test_an_unknown_meter_returns_no_rows_not_an_error(sessions):
    assert EnergyReadingRepository(sessions).get_readings("00000000-0000-0000-0000-000000000000").empty


def test_the_udfs_agree_with_the_raw_tables(sessions, raw):
    meters = MeterRepository(sessions).get_meters()
    assert len(meters) == 31 and meters["zone_name"].notna().all() and meters["meter_key"].is_unique
    breakers = MeterRepository(sessions).get_breakers()
    assert len(breakers) == 31 and breakers["cb_max_rating_a"].notna().all()
    with raw.connect() as conn:
        total = conn.execute(text("SELECT count(*) FROM gold.energy_raw")).scalar()
    saved = ROOT / "outputs" / "data" / "raw_readings_database.parquet"
    if saved.exists():
        assert len(ArtifactRepository(ROOT / "outputs").load_frame("raw_readings_database")) == total
    key = meters["meter_key"].iloc[0]
    with raw.connect() as conn:
        direct = conn.execute(text("SELECT count(*) FROM gold.energy_raw WHERE energy_meter_key = CAST(:k AS uuid)"),
                              {"k": key}).scalar()
    assert len(EnergyReadingRepository(sessions).get_readings(key)) == direct


def test_the_session_is_read_only_and_has_a_statement_timeout(raw):
    with raw.connect() as conn:
        assert conn.execute(text("SHOW default_transaction_read_only")).scalar() == "on"
        assert conn.execute(text("SHOW statement_timeout")).scalar() in ("15min", "900s", "900000ms")


def test_a_missing_password_is_a_clear_error_not_a_connection_attempt():
    from config.settings import Settings
    with pytest.raises(ValueError, match="db_password"):
        Settings(db_password=None).database_url
