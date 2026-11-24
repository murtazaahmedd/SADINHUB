# SADIN Energy Forecasting & Anomaly Detection

Daily kWh forecasting, bill estimation and anomaly alerts for a 31-meter facility, built as a layered
Python service on top of a PostgreSQL medallion warehouse.

**Facility day-ahead accuracy: 95.3%** (100 − WAPE, every morning of 2026 replayed with only the data
available that morning). All 31 meters clear the 80% target at a 1-day horizon.

---

## What it does

- **Forecasts** daily kWh for every meter, zone and the whole facility, out to 31 December.
- **Proves its accuracy** by replaying every morning of the year with walk-forward retraining, so a
  replayed forecast never sees its own future.
- **Estimates the bill** per meter and billing cycle from the tariff, breaker fees and VAT.
- **Flags unusual consumption** with a z-score rule and an Isolation Forest, groups repeat alerts into
  incidents ranked by kWh and cost impact, and catches overnight base-load faults a daily total hides.
- Runs as a **command-line pipeline**, a **FastAPI service** and an **offline HTML dashboard**.

## Architecture

```text
 Excel telemetry exports ─┐
                          ├─► bronze (raw) ─► silver (validated 15-min intervals) ─► gold (meter, zone, energy)
 Meter warehouse dump ────┘                                  PostgreSQL, read only through api.* SQL functions
                                                                              │
 Open-Meteo weather ──────────────────────────────────────────────────────────┤
                                                                              ▼
                     Stage 1: Huber regression on physical drivers (temperature hinge curve, weather
                              anomaly, weekday, Ramadan/Eid/national days, momentum), per horizon band
                     Stage 2: LightGBM on the stage-1 residuals (0–6 days ahead)
                                                                              │
                  ┌──────────────────────────┬────────────────────────────────┼──────────────────────┐
                  ▼                          ▼                                ▼                      ▼
          daily forecast              accuracy replay                  bill forecast        alerts & incidents
          (meter/zone/facility)       (walk-forward)                   (tariff + VAT)       (z-score + Isolation Forest)
                  └──────────────────────────┴───────────────┬────────────────┴──────────────────────┘
                                                             ▼
                                         FastAPI  ·  CLI  ·  HTML dashboard
```

## Results

| Level | 1 day | 7 days | 30 days | Calendar month |
|---|---|---|---|---|
| Facility | **95.3%** | 93.9% | 90.3% | 90.5% |
| Zone (21) | 90.8% | 88.8% | 86.3% | 84.2% |
| Meter (31) | 89.8% | 87.7% | 84.9% | 82.7% |

The model beats a naive "last 7 days held flat" baseline at every level. Full write-up, including known
weaknesses: [VALIDATION_REPORT.md](ML_Work2/documentation/VALIDATION_REPORT.md).

## Tech stack

Python · pandas · NumPy (custom Huber IRLS regression) · scikit-learn (IsolationForest) · LightGBM · PostgreSQL (PL/pgSQL `api.*`
functions) · FastAPI · Pydantic · pytest

## Engineering highlights

- **Clean layering with dependency inversion:** routers → services → repositories, each behind an
  interface and wired in one composition root. Architecture tests fail the build if a layer is skipped.
- **Read-only data access:** every query is a versioned SQL function; the Python session rejects any
  statement that is not a read.
- **No hard-coded years:** validation windows, forecast end, billing cycles and Ramadan/Eid dates are all
  derived from the data and the Umm al-Qura calendar.
- **Adversarial tests:** refits on corrupted real data, synthetic meters with known answers, injected
  faults (×0.4, ×1.6, ×3, zero) to measure alert recall.

## Run it

Setup, database restore, CLI flags and the API reference are in
[ML_Work2/README.md](ML_Work2/README.md). In short:

```bash
cd ML_Work2
conda env create -f environment.yml && conda activate ml_work2
python run_pipeline.py --refresh      # replay the year, forecast to 31 Dec, build the dashboard
uvicorn main.app:app --port 8000      # API docs at http://localhost:8000/docs
```

## Documentation

- [How the forecast works](ML_Work2/documentation/HOW_THE_FORECAST_WORKS.md)
- [How the alerts work](ML_Work2/documentation/HOW_THE_ALERTS_WORK.md)
- [Validation report](ML_Work2/documentation/VALIDATION_REPORT.md)

---

Built by [Murtaza Ahmed](https://github.com/murtazaahmedd) · [Portfolio](https://murtazaahmedd.github.io)
