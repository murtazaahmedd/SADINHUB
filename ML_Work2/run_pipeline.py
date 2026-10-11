"""Run the whole SADIN forecast: data, 2026 validation, forecast, dashboard.

Usage (from this folder):
    .venv\\Scripts\\python.exe run_pipeline.py                  # use cached readings
    .venv\\Scripts\\python.exe run_pipeline.py --refresh        # re-read database and Excel
    .venv\\Scripts\\python.exe run_pipeline.py --dashboard-only # rebuild the page from the last run

The database password is read from the SADINHUB_PASSWORD (or PGPASSWORD)
environment variable, or asked for interactively. It is never saved.
New Excel exports go in input_excel/; run with --refresh to pick them up.
"""
import sys
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

from forecasting.dashboard import build_dashboard  # noqa: E402
from forecasting.pipeline import run, saved_results  # noqa: E402

if __name__ == "__main__":
    if "--dashboard-only" in sys.argv:
        results = saved_results()
    else:
        results = run(refresh_sources="--refresh" in sys.argv)
    path = build_dashboard(*results)
    print(f"Dashboard: {path}")
