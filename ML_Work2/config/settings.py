"""Runtime settings: folders, database connection and the dates derived from the data."""
import os
from datetime import date
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict

from utilities.guard import require, traced

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseModel):
    """
    #Feature: Configuration
    #Scenario: One immutable object holds every folder and connection value
    #Complexity: Time O(1) - constant

    Folders, database connection and API key. Nothing about the year is stored here.
    """

    model_config = ConfigDict(frozen=True)

    root: Path = ROOT
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "sadinhub"
    db_user: str = "postgres"
    db_password: Optional[str] = None
    api_key: Optional[str] = None

    @property
    def input_excel(self) -> Path:
        """Folder of telemetry Excel exports."""
        return self.root / "input_excel"

    @property
    def outputs(self) -> Path:
        """Root of every saved result."""
        return self.root / "outputs"

    @property
    def database_url(self) -> str:
        """SQLAlchemy URL for the read-only warehouse connection."""
        password = require("Settings.database_url", "db_password", self.db_password)
        return f"postgresql+psycopg://{self.db_user}:{password}@{self.db_host}:{self.db_port}/{self.db_name}"


@traced
def load_settings() -> Settings:
    """
    #Feature: Configuration
    #Scenario: Build Settings from environment variables, never from files
    #Complexity: Time O(1) - constant

    Read SADINHUB_HOST/PORT/NAME/USER/PASSWORD (or PGPASSWORD) and SADIN_API_KEY.

    Returns:
        A frozen Settings object.

    Raises:
        ValueError: When SADINHUB_PORT is not an integer.
    """
    env = os.environ
    values = {
        "db_host": env.get("SADINHUB_HOST", "localhost"),
        "db_port": int(env.get("SADINHUB_PORT", "5432")),
        "db_name": env.get("SADINHUB_NAME", "sadinhub"),
        "db_user": env.get("SADINHUB_USER", "postgres"),
        "db_password": env.get("SADINHUB_PASSWORD") or env.get("PGPASSWORD"),
        "api_key": env.get("SADIN_API_KEY"),
    }
    return Settings(**values)


@traced
def forecast_end_for(last_data_day: date) -> date:
    """
    #Feature: Year-agnostic dates
    #Scenario: The forecast runs to 31 December of the year the data ends in
    #Complexity: Time O(1) - constant

    Args:
        last_data_day: Last day with data.

    Returns:
        31 December of that year.

    Raises:
        ValueError: When last_data_day is None.
    """
    require("forecast_end_for", "last_data_day", last_data_day)
    return date(last_data_day.year, 12, 31)


@traced
def validation_start_for(last_data_day: date) -> date:
    """
    #Feature: Year-agnostic dates
    #Scenario: Every-morning replay starts on 1 January of the year the data ends in
    #Complexity: Time O(1) - constant

    Args:
        last_data_day: Last day with data.

    Returns:
        1 January of that year.

    Raises:
        ValueError: When last_data_day is None.
    """
    require("validation_start_for", "last_data_day", last_data_day)
    return date(last_data_day.year, 1, 1)
