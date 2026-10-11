# SADIN Facility – kWh Forecasting (ML_Work2)

Forecasts daily kWh for every meter, zone and the whole facility, checks every
forecast made in 2026 against the gold-validated actuals, and forecasts ahead
to 31 December 2026.

**Open the dashboard:** double-click `Open_Dashboard.cmd`
(or open `dashboard/SADIN_Energy_Forecast_Dashboard.html`; it works offline).

**How it works and how accurate it is:** `documentation/HOW_THE_FORECAST_WORKS.md`

## Folders

| Folder | What is in it |
|---|---|
| `dashboard/` | The dashboard (one HTML file, predictions and accuracy only) |
| `documentation/` | Plain-language explanation of the model and its 2026 results |
| `forecasting/` | The code (data, model, validation, dashboard builder) |
| `input_excel/` | Telemetry exports after the database (12 Jul – 3 Oct 2026) |
| `outputs/data/` | Daily kWh per meter, weather history, reading caches |
| `outputs/validation/` | Every 2026 forecast vs actual, accuracy tables by month |
| `outputs/forecasts/` | Forecast to 31 Dec 2026 (meter, zone, facility, monthly outlook) |
| `tests/` | Automatic checks (no future data used, gold reproduced exactly) |
| `archive_gpt_version/` | The previous version, kept unchanged for reference |
| `reference_project/` | Copy of the original project (database scripts, reports) |

## Run it again (e.g. after new Excel exports)

1. Put the new exports in `input_excel/` (file names like the current ones).
2. Set the database password for this window only, then run:

```bash
set SADINHUB_PASSWORD=your_password
.venv\Scripts\python.exe run_pipeline.py --refresh
```

Without `--refresh` the cached readings are reused (about 5 minutes).
The password is never written to any file; if it is not set, the script asks.
To rebuild only the dashboard page from the last run (seconds, no retraining):
`.venv\Scripts\python.exe run_pipeline.py --dashboard-only`

3. Check: `.venv\Scripts\python.exe -m pytest tests -q`

## Key outputs

* `outputs/validation/accuracy_summary_2026.csv` – accuracy per level and horizon
* `outputs/validation/accuracy_by_month_<meter|zone|facility>_<horizon>.csv` – month tables
* `outputs/validation/validation_detail_2026.csv` – every single forecast vs actual
* `outputs/validation/every_morning_forecast_2026.parquet` – the full forecast made on every morning of 2026 (feeds the dashboard's date explorer)
* `outputs/validation/day_ahead_forecast_vs_actual_2026.csv` – daily forecast next to actual
* `outputs/forecasts/forecast_daily_meters_to_2026-12-31.csv` – the forward forecast
* `outputs/forecasts/monthly_outlook_2026_actual_plus_forecast.csv` – 2026 by month
* `outputs/forecasts/bill_forecast_2026_by_meter.csv` – predicted bill (SAR) per meter per 30-day cycle (tariff in `forecasting/billing.py`)
