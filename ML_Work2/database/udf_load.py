"""Admin script: create the api.* functions in the warehouse.

This is the only code that writes to the database. The pipeline itself stays
read-only and only calls these functions.

    python -m database.udf_load          # needs SADINHUB_PASSWORD
"""
from pathlib import Path
from typing import List

from sqlalchemy import create_engine

from config.settings import Settings, load_settings
from utilities.guard import require, traced

UDF_DIR = Path(__file__).resolve().parents[1] / "sql" / "udf" / "api"

# Every function the repositories call. Register new ones here.
UDF_FILES: List[str] = [
    "get_energy_meters.sql",
    "get_energy_readings.sql",
    "get_meter_breakers.sql",
    "get_gold_daily_consumption.sql",
]


@traced
def load_udfs(settings: Settings) -> List[str]:
    """
    #Feature: Database setup
    #Scenario: Create or replace every registered api.* function in one transaction
    #Complexity: Time O(f) - registered functions

    Args:
        settings: Connection settings (a write-capable user).

    Returns:
        The names of the files that were applied.

    Raises:
        ValueError: When settings is None or the password is missing.
        FileNotFoundError: When a registered file is missing.
    """
    require("load_udfs", "settings", settings)
    engine = create_engine(settings.database_url)
    applied: List[str] = []
    with engine.begin() as conn:
        for name in UDF_FILES:
            path = UDF_DIR / name
            if not path.exists():
                raise FileNotFoundError(f"[load_udfs] Registered UDF file not found: {path}. "
                                        f"Invariant violated: every UDF_FILES entry exists. Fix: add the file.")
            conn.exec_driver_sql(path.read_text(encoding="utf-8"))
            applied.append(name)
    return applied


if __name__ == "__main__":
    for applied_name in load_udfs(load_settings()):
        print(f"applied {applied_name}")
