# SADIN Facility - kWh Forecasting (ML_Work2)

Forecasts daily kWh for every meter, zone and the whole facility; replays every morning of the year to
prove how accurate it was; forecasts to 31 December from **all** the data available; estimates the bill;
and flags unusual meter-days. It runs as a command line tool and as an HTTP API.

Nothing is tied to a year: the validation year, the forecast end and the billing cycles are derived from the
last day of data, and Ramadan / Eid dates are derived from the Umm al-Qura calendar.

Validation results and known weaknesses: [documentation/VALIDATION_REPORT.md](documentation/VALIDATION_REPORT.md).
How the model works: [documentation/HOW_THE_FORECAST_WORKS.md](documentation/HOW_THE_FORECAST_WORKS.md).
How the alerts work: [documentation/HOW_THE_ALERTS_WORK.md](documentation/HOW_THE_ALERTS_WORK.md).

---

## 1. Make it runnable (first time)

### 1.1 Python environment

```bash
conda env create -f environment.yml          # or: python -m venv .venv && pip install -r requirements.txt
conda activate ml_work2
```

### 1.2 The warehouse (local PostgreSQL, database `sadinhub`)

The pipeline reads meter readings from PostgreSQL through four read-only functions (`api.*`). Host, port,
database and user default to `localhost` / `5432` / `sadinhub` / `postgres` and can be overridden with
`SADINHUB_HOST`, `SADINHUB_PORT`, `SADINHUB_NAME`, `SADINHUB_USER`. The password is read from `SADINHUB_PASSWORD`
(or `PGPASSWORD`), or asked for interactively; it is never written to a file.

```bash
createdb -h localhost -U postgres sadinhub
pg_restore -h localhost -U postgres -d sadinhub --no-owner -j 4 DATABASE_BACKUP.dump     # raw source data (schema public)
psql -h localhost -U postgres -d sadinhub -c "ALTER SCHEMA public RENAME TO bronze; CREATE SCHEMA public;"
psql -h localhost -U postgres -d sadinhub -f rebuild_gold_from_bronze.sql                # gold.meter / gold.zone / gold.energy_raw
export SADINHUB_PASSWORD='...'
python -m database.udf_load                                                              # creates the api.* functions
```

`rebuild_gold_from_bronze.sql` rebuilds the three gold objects the pipeline reads from the raw data. If you have the
original gold and silver layers, skip it. It also creates a **stand-in** `silver.dim_meter_cb_rating` (breaker ratings from the
billing accounts); only the SAR bill forecast depends on it.

### 1.3 The Excel exports

Put the telemetry exports (`Telemetry_Dumps_..._ZN.<zone>-PB[.<n>]_child_<n>_ch3pm_<date>.xlsx`, columns `timestamp` in
epoch ms and `kWh`) in `input_excel/`. Each file is matched to its meter by **register continuity**, not by its name, so a
misnamed file is still placed correctly and a file that continues no meter is refused. With no files the pipeline runs on
warehouse data only.

### 1.4 Run it

```bash
python run_pipeline.py --refresh        # re-read warehouse + Excel, replay the year, forecast to 31 Dec, build the dashboard (about 15 min)
python run_pipeline.py                  # reuse the saved readings (about 7 min)
python run_pipeline.py --alerts         # recompute the alerts and rebuild the dashboard's Alerts tab (about 20 s)
python run_pipeline.py --dashboard-only # rebuild the page from the last run (seconds)
./open_dashboard.sh                     # open dashboard/SADIN_Energy_Forecast_Dashboard.html (Windows: Open_Dashboard.cmd)
```

Order after new data: `--refresh`, then `--alerts`. The final forecast is always trained on every complete day available.

### 1.5 Run the API

```bash
uvicorn main.app:app --port 8000        # interactive docs at http://localhost:8000/docs
export SADIN_API_KEY='...'              # optional: then every endpoint except /health needs an X-API-Key header
```

| Endpoint | What it does |
|---|---|
| `GET /health` | status and which saved results exist |
| `POST /pipeline/run` `{"refresh_sources": false}` | full run (minutes) |
| `POST /pipeline/dashboard` | rebuild the page from the last run |
| `GET /forecast/{meters\|zones\|facility}?start=&end=&page=&limit=` | the saved daily forecast |
| `GET /accuracy/summary`, `GET /accuracy/by-month?level=&horizon=` | the saved accuracy tables |
| `POST /alerts/run`, `GET /alerts?period=&detector=&meter=`, `GET /alerts/data-problems` | daily alerts and data problems |
| `GET /alerts/incidents?active_only=`, `GET /alerts/night-load` | repeat alerts grouped into incidents (ranked by impact), and night base-load alerts |
| `GET /bills?cycle=` | the bill forecast |

Reads before the first run return `409` with an explanation. Errors are always JSON `{"detail": "..."}`.

---

## 2. Repository structure

Dependencies flow **downward only**; each layer talks to the next through an interface (constructor injection).

```text
routers/        HTTP endpoints. @http_error_handler + @enforce + schema binding. No logic.
decorators/     enforce (API key + permission check), http_error_handler (maps errors to HTTP statuses)
schemas/        Pydantic request/response models, one folder per domain
services/       business logic and orchestration, one folder per domain + container.py (lazy wiring)
                  data, forecast, accuracy, billing, anomaly, dashboard, pipeline
interfaces/     abstract base classes for every repository and service (dependency inversion)
repositories/   all I/O: warehouse (api.* UDF calls), Excel exports, weather API, saved results + container.py
database/       read-only session factory with a statement guard; udf_load.py (the only code that writes to the DB)
sql/udf/api/    the api.* functions (CTEs). No SQL is written anywhere else.
models/         the ML: forecasting/ (features, temperature response, forecaster) and anomaly/ (features, detectors)
utilities/      pure helpers: gold rule, daily table, scoring, billing rules, calendar, Excel matching, dashboard data
constants/      every literal: tariff, model parameters, windows, thresholds, known events, file names
config/         settings (folders, connection) and the dates derived from the data
errors/         error taxonomy (Errors.HTTP.*, Errors.Misc.*)
main/           app.py (FastAPI + composition root), cli.py (command line), dependencies.py
tests/          behaviour, architecture and adversarial tests
documentation/  model, alert and validation write-ups
outputs/        everything the pipeline saves (no year in any file name)
input_excel/    the telemetry exports
dashboard/      the offline HTML dashboard
reference_project/, archive_gpt_version/, anomaly_detection/_backup_before_alerts_tab/   original material, kept unchanged
```

### The ML, in one paragraph

Each day's forecast is the meter's recent level (last 7 days) multiplied by an expected change. Stage 1 is a robust (Huber)
linear model on physical drivers (seasonal temperature from a per-meter hinge curve, recent weather anomaly, weekday pattern,
Ramadan/Eid/national days, momentum, and yesterday for the next days), fitted separately for each band of days ahead. Stage 2 is a
small LightGBM model on what stage 1 leaves unexplained, used only for 0-6 days ahead. Models are retrained every month on data
before that month, so a replayed forecast never sees its own future. Alerts compare each day with that morning's forecast using a
simple z-score rule and an Isolation Forest, both gated on a material difference and trustworthy readings. The Isolation Forest trains on
every replayed day before April 2026 (from April 2025, a full seasonal cycle) and its alert level is a fixed 3% of the checked days of April-August. Repeat alerts are grouped into incidents ranked by kWh and SAR impact, and a separate night base-load detector catches small loads left running overnight that a daily total hides.

### Where to change things

| To change | Edit |
|---|---|
| Tariff, breaker fees, VAT | `constants/tariff.py` |
| Model parameters, windows, horizons | `constants/forecasting.py` |
| Alert thresholds, period plan, confirmed events | `constants/anomaly.py`, `constants/known_events.py` |
| Data rules (startup exclusions, gap length, plausibility cap) | `constants/data_rules.py` |
| Weekend days, fixed holidays | `constants/calendar.py` |
| Add an endpoint | schema, service interface, service, `services/container.py`, router (the architecture tests tell you what is missing) |
| Add a database query | a new `sql/udf/api/*.sql`, register it in `database/udf_load.py`, call it from a repository |

### Outputs (`outputs/`)

* `validation/` accuracy tables, `every_morning_forecast.parquet`, `validation_detail.csv`, `day_ahead_forecast_vs_actual.csv`, learned weights
* `forecasts/` daily forecast by meter, zone and facility, monthly outlook, `bill_forecast_by_meter.csv`, `forecast_info.json` (what the model was trained on)
* `alerts/` consumption alerts, `incidents.csv`, `night_load_alerts.csv`, data-problem alerts, `evaluation.json`
* `data/` daily table, validated intervals, readings caches, weather, Excel identity audit

---

## 3. Tests and validation

```bash
python -m pytest -q -m "not slow"        # about 1 minute: unit, architecture, adversarial, validation of the saved results
python -m pytest -q                      # everything, including refits on corrupted real data (about 2.5 minutes)
```

175 tests in `tests/`. The database tests need `SADINHUB_PASSWORD`; tests on saved results skip until a pipeline run exists.

| File | Purpose |
|---|---|
| `test_architecture.py` | layers only depend downward, no inline SQL in repositories, every package exports explicitly, every endpoint is marked, no function over 30 lines of code |
| `test_forecasting.py` | no-leakage proofs, training cut-off, derived calendar, bill rules, horizon guard |
| `test_adversarial_ingestion.py` | glitchy registers, resets, jumps, outages, swapped or corrupt Excel files |
| `test_adversarial_model.py` | synthetic meters with known answers; refits on spiked, outage-hit and heatwave data |
| `test_validation_outputs.py` | data integrity, exact energy accounting, forecast sanity, independent accuracy, naive baselines |
| `test_adversarial_anomaly.py`, `test_anomaly.py`, `test_adversarial_alert_types.py` | injected faults, noise, data starvation, incidents, night load |
| `test_adversarial_api.py`, `test_api.py` | hostile parameters, API-key near misses, concurrency |
| `test_adversarial_database.py` | read-only enforcement, SQL injection, UDF versus raw tables |
| `test_real_bills.py` | energy accounting and data-problem alerts against 28 real utility bills (needs the database) |

## 4. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `409 ... has not been created yet` | the pipeline has not run: `python run_pipeline.py --refresh` |
| `Warehouse call failed ... run python -m database.udf_load` | the api.* functions are missing, or the password is wrong |
| `No meter continues the register of <file>` | an export does not follow on from the warehouse data (wrong period or facility) |
| `The forecast needs N days ahead but the model is trained for 180` | the data ends more than 180 days before 31 Dec: load newer data, or raise `MAX_HORIZON_DAYS` |
| `development period has no scored meter-days` | run the forecasting pipeline over the full year before `--alerts` |
| `Refused statement ...` | something tried to run SQL other than `SELECT * FROM api.<function>(...)` through a pipeline session |
