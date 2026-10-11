"""Run the whole SADIN forecast: data, validation of the year, forecast, dashboard.

Usage (from this folder):
    python run_pipeline.py                   # use saved readings
    python run_pipeline.py --refresh         # re-read the warehouse and the Excel exports
    python run_pipeline.py --dashboard-only  # rebuild the page from the last run
    python run_pipeline.py --alerts          # recompute the alerts and rebuild the page

Start the API instead with:  uvicorn main.app:app
"""
import sys

from main.cli import main

if __name__ == "__main__":
    sys.exit(main())
