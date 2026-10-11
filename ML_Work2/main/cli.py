"""Command line: run the pipeline or the alerts without starting the API.

    python run_pipeline.py                   # use saved readings
    python run_pipeline.py --refresh         # re-read the warehouse and the Excel exports
    python run_pipeline.py --dashboard-only  # rebuild the page from the last run
    python run_pipeline.py --alerts          # recompute the alerts and rebuild the page

The database password is read from SADINHUB_PASSWORD (or PGPASSWORD), or asked
for interactively. It is never saved.
"""
import getpass
import sys
import warnings
from typing import List, Optional

from config.settings import Settings, load_settings
from main.app import build_service_container
from utilities.guard import traced


@traced
def _with_password(settings: Settings, needs_database: bool) -> Settings:
    """
    #Feature: Command line
    #Scenario: Ask for the database password only when a database read is about to happen
    #Complexity: Time O(1) - constant

    Args:
        settings: Settings from the environment.
        needs_database: Whether this command reads the warehouse.

    Returns:
        Settings with the password filled in (when needed).

    Raises:
        ValueError: When settings is None.
    """
    if settings.db_password or not needs_database:
        return settings
    return settings.model_copy(update={"db_password": getpass.getpass("SADINhub database password: ")})


@traced
def main(argv: Optional[List[str]] = None) -> int:
    """
    #Feature: Command line
    #Scenario: Dispatch one command
    #Complexity: Time O(months * fit) - one model per month

    Args:
        argv: Command-line arguments (defaults to sys.argv).

    Returns:
        Exit code 0.

    Raises:
        Errors.Misc.DataNotReady: When a command needs results that do not exist yet.
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    args = sys.argv[1:] if argv is None else argv
    settings = _with_password(load_settings(), needs_database="--dashboard-only" not in args and "--alerts" not in args)
    services = build_service_container(settings)
    if "--alerts" in args:
        print(services.anomaly.run())
    elif "--dashboard-only" in args:
        print(f"Dashboard: {services.pipeline.build_dashboard()}")
    else:
        results = services.pipeline.run(refresh_sources="--refresh" in args)
        print(f"Dashboard: {services.pipeline.build_dashboard(results)}")
    return 0
