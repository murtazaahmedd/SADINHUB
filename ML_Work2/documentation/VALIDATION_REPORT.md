# Validation and adversarial testing report

Run on 2026-10-11 against all data now available: warehouse readings to 2026-07-11 plus the 31 Excel
exports (12 Jul - 3 Oct 2026), so every meter reports through **2026-10-02**. The final forecast model is
trained on **all** of it (1,053,600 training examples, trained through 2026-10-02) and forecasts 3 Oct - 31 Dec 2026.

**Verdict.** The pipeline is correct, reproducible and leak-free, and it beats a naive baseline at every
level and horizon. It is accurate for the facility and for most meters at short horizons. Four
implementation defects were found and fixed during this work. Three modelling weaknesses remain open (section 8).

Everything below is reproducible with `python -m pytest -q` (175 tests: 174 pass, 1 skips because
`gold.energy_consumption_validated` is not in the local database).

## 1. The restructure changed no result

The original code and the new layered code were both run from scratch on the same data and every output file was compared.

| Compared | Result |
|---|---|
| 28 of 30 output files (daily table, every-morning forecasts, all accuracy tables, year-end forecast, monthly outlook, weights, long-range accuracy) | bit-for-bit identical, maximum difference 0 |
| `validated_intervals` | identical (the "mismatch" was `NaT` vs `NaT` in each meter's first reading) |
| `bill_forecast` | identical once the original code's bill step was run on the same data |
| Dashboard data blocks (accuracy, calendar months, dates, explorer, long range, outlook, series, units) | identical |
| Dashboard `meta` / `bills` | differ only by the two intended additions (`year`, `long_range_since`) and the wording "runs into next year" |

Replacing the hand-typed 2025-2027 holiday tables with dates derived from the Umm al-Qura calendar also
changed nothing: every accuracy figure and the year-end forecast total are identical.

## 2. Ingestion (Excel exports)

* All 31 exports were matched to the right meter, including the two with misleading names, by register continuity
  (`outputs/data/excel_meter_identity.json`). 903,412 Excel readings; no reading appears in both sources.
* At the database/Excel handover every register is continuous (never goes backwards) and the implied power is below the 500 kW cap.

Adversarial tests (`tests/test_adversarial_ingestion.py`, 18): swapped file names are resolved by register,
an export that continues no meter is refused (never guessed), an impossible rate is refused, assignment does not
depend on file order, garbage cells and an empty or missing folder are survived, overlapping sources never double count.

## 3. Data integrity of the saved results

* **Exact energy accounting for all 31 meters:** booked kWh = net register rise + ignored decrements - ignored jumps, to 1e-5 kWh.
  This caught nothing wrong but pins the rule down: the pipeline never invents energy.
* The ZN.13-PB.2 register reset on 2025-02-05 (3,723.5 to 778.5 kWh) books exactly 0 kWh, as designed.
* Synthetic registers (random gaps up to multi-day outages, glitches, counter resets, impossible jumps, NaNs, startup readings):
  consumption is never negative, isolated spikes and dips book nothing, long gaps keep their energy.
* No meter-day exceeds the physical maximum (500 kW x 24 h); every meter has a zone; 21 zones, 31 meters.

## 4. Model correctness

| Check | Result |
|---|---|
| Leakage: scramble every actual, flag and temperature from the issue day onward | features identical (5 dates tested) |
| Label-leak: retrain with everything after the cutoff scrambled (incl. +70 C weather) | learned weights identical |
| Reproducibility: refit and compare with the saved walk-forward output | 0.000000 kWh difference |
| Independent recomputation of the reported 1-day accuracy from the raw daily table | matches to 0.01 |
| Synthetic meters with known answers: constant, all-zero, x10 scaled, 40-day history, 20-day trailing outage | constant stays constant (+-8%), dead meter stays at 0, x10 history gives x10 forecast (+-15%), outage falls back to last trusted level |
| Robust regression with 5% gross outliers | coefficient error 0.001 vs 0.0135 for ordinary least squares |
| Temperature hinge fit on synthetic data | recovers true slopes to 0.001 |
| Forecast file | 31 meters x every day to 31 Dec, finite, non-negative, below 500 kW, zone and facility are exact sums of meters |
| Forecast level | first 30 days average 4,549 kWh/day; September 2026 ran 6,005 and October 2025 ran 4,077, in line with the seasonal drop |

## 5. Accuracy of the 2026 replay (every morning replayed with data available that morning)

Accuracy = 100 - WAPE. Scored on measurable days only (8.8% of one-day meter periods are excluded for outages and gaps).

| Level | 1 day | 7 days | 30 days | Calendar month |
|---|---|---|---|---|
| Facility | 95.3 | 93.9 | 90.3 | 90.5 |
| Zone (21) | 90.8 | 88.8 | 86.3 | 84.2 |
| Meter (31) | 89.8 | 87.7 | 84.9 | 82.7 |

Meters at or above the 80% target: 31/31 at 1 day, 28/31 at 7 days, 26/31 at 30 days, 23/31 for whole months.
Facility accuracy by distance ahead (a forecast issued on the 1st of each month): **90.4% (days 1-30), 86.3% (31-60), 80.1% (61-90).**

**Against a naive baseline ("last 7 days held flat")** the model wins everywhere:

| Level | 1 day | 7 days | 30 days |
|---|---|---|---|
| Meter | 89.8 vs 85.3 (+4.5) | 87.7 vs 86.3 (+1.3) | 84.9 vs 82.4 (+2.5) |
| Zone | 90.8 vs 86.8 (+4.0) | 88.8 vs 87.7 (+1.1) | 86.3 vs 83.8 (+2.5) |
| Facility | 95.3 vs 91.8 (+3.4) | 93.9 vs 92.2 (+1.7) | 90.3 vs 86.6 (+3.7) |

## 6. Alerts (anomaly detection)

All four periods exist (development Apr 2025 - Mar 2026, validation Apr-Aug 2026, final test Sep, latest Oct), so the full evaluation runs.
There are no labelled faults, so faults were injected into 2,727 real meter-days (validation to latest, forecast above 20 kWh):

| Injection | Flagged | Rule | ML |
|---|---|---|---|
| actual x 1.6 | 88.9% | 88.6% | 67.1% |
| actual x 0.4 | 94.3% | 91.7% | 90.4% |
| actual x 3.0 | 99.2% | 99.2% | 99.0% |
| actual = 0 | 99.9% | 98.6% | 99.9% |
| +-5% noise (should not alert) | 6.1% / 5.3% | | natural alert rate is about 5% |

Injecting nothing creates no new alerts; flagged direction always matches the sign of the difference. Known events are found
(ZN.16 switch-on and ZN.8-PB.2 new load first alerted 2026-06-27, ZN.13-PB.1 drop 2026-09-07, the February site outage is reported as a
data problem). Final-test agreement: 22 both, 35 rule only, 5 ML only. (These injection figures are on the September-October days, the period not used for calibration.)

### 6.1 Two changes made to the detector, and what they did

1. **Trained on a full seasonal cycle.** Same-day forecasts were replayed back to 1 Apr 2025 (each early morning forecast by a model
   trained only on data before that month), so the Isolation Forest trains on **11,509 meter-days over 15 months instead of 4,555 over 5.5 months.**
2. **A fixed alert level.** The ML cut-off flags a fixed **3%** of the validation period's checked days (about 6 alerts a week, a staffing
   choice) instead of being set to match the number of alerts the simple rule raised. It no longer depends on the rule; a test proves
   that doubling the rule's threshold leaves the ML cut-off unchanged.

| Measured | Before | After |
|---|---|---|
| ML training rows | 4,555 | 11,509 |
| ML alert rate, validation (Jul-Aug) | 3.0% (matched to the rule) | 3.0% (fixed target) |
| ML alert rate, final test (Sep) | 7.0% | 6.8% |
| Fault detection x1.6 / x0.4 / x3 / zero | 94.1 / 99.6 / 99.7 / 100% | 94.5 / 99.5 / 99.7 / 100% |
| Known events found | all | all, same first-alert dates |

**Honest outcome: no measurable improvement in detection.** The detector is more robust in principle (it has seen every season in the
data) and its alert level no longer depends on the rule, but the results above are the same. The September alert rate is more than
double the target, and the explanation is in the data, not in the model: the simple rule's alert rate also doubled from 3.0% (Jul-Aug) to
6.3% (Sep), because July-August was the calmest period (median forecast error 5.0% of the expected kWh against 6.2% in September and 8.0%
across the development period). A rate fixed on the quietest window therefore under-states the normal alert load in other months.

### 6.2 Round two: incidents, night load, a longer calibration window (and one idea dropped)

| Change | Measured result | Kept? |
|---|---|---|
| **Calibrate the ML alert level on a longer, typical window** (Apr-Aug, five months) instead of the calmest two (Jul-Aug) | ML alert rate in the untouched September test **6.8% -> 3.0%** (target 3%), ML alerts 63 -> 27, all alerts 73 -> 62. The cost: injected x1.6 faults caught 92.3% -> 88.9% and x0.4 faults 98.9% -> 94.3% on the same September-October days | **Yes**, a deliberate trade: the ML detector now delivers the alert load it was set to, and does so far more efficiently per alert. The target is `ML_TARGET_ALERT_RATE` |
| **Group repeat alerts into incidents, ranked by kWh and SAR impact** | 832 alerts -> 707 incidents (15% fewer); every alert sits in exactly one incident; ZN.16 (about 1,590 kWh) and ZN.8-PB.2 (about 470 kWh) are the top two; 10 incidents are still running at the end of the data | **Yes.** Most alerts are isolated days, so the volume saving is modest; the ranking is the benefit |
| **Night base-load detector** (midnight-5 am average kW against the meter's own recent nights, 5 usual variations and 1 kW) | Faults added to the night hours of 912 real days: +2 kW found by the daily detectors 11% / by night 44% / either 49%; +5 kW 28% / 62% / 67%; +10 kW 47% / 72% / 78%; +20 kW 87% / 80% / 95%. Natural alert rate 2-4% of nights (about 5 a week) | **Yes.** It finds a real class of fault the daily detectors miss |
| **Level-change detector** (CUSUM on a meter's share of the facility, to date permanent changes) | CUSUM that re-fires: 228 episodes in 22 months. CUSUM that latches until it resets: 31 of 66 episodes never close, some for 595 days, because real share noise exceeds the noise floor. Ratio against the previous month: 3 a month but it missed ZN.16's switch-on and ZN.13-PB.1's drop | **No. Removed.** It did not meet the bar of beating the existing detectors on the known events without adding noise. Needs labelled changes to tune |

Weaker numbers for the ML half are real: the synthetic +-60% check now reads 84.4% / 90.0% (was 87.8% / 94.7%), because the ML detector is stricter.
The simple rule is unchanged and catches 88-99% of the injected faults on its own.

### 6.3 Tests on real ground truth

There are **no labelled consumption faults** anywhere in the data. The source system's alarm tables are empty except 224,764
power-factor threshold alarms (April-July 2026), which concern a different quantity. What does exist, and was used, is the
facility's own record of **28 real utility bills** (14 May - 12 Jun 2025): the energy the grid billed, and the building system's
own figure for the same period. They test the energy accounting and the data-problem detector, not the consumption detectors.

| Real test | Result |
|---|---|
| Our energy against the building system's own figure, 18 meters with every day trusted | equal to 0.0% (median) for 16 of 18 |
| The same against the utility, the 13 clean meters within 2% | all within -1.9% .. +0.1%; the utility reads about 0.6% more than the meters |
| 6 meters with a ~3.6 day outage | time-spread booking is within 3% of the utility bill (-2.9% .. +1.0%); gold-style booking is **29-48% above** it, and the building system's own figure is 10-13% below it |
| Data-problem detector on the 11 meters that had missing days in the bill window | an episode overlapped the window for **11 of 11** |
| Meters with 16-27 missing days | spread booking is not reliable over long outages: ZN.17 -7.5%, ZN.10-PB.2 -0.1%, ZN.9 +83% against the bill |

**Six meters disagree with the utility even though their data is complete and consistent**: ZN.4-PB.1 (+104%), ZN.10-PB.1 (-42%),
ZN.11-PB.2 (-50%, exactly half), ZN.12 (-87%), ZN.8-PB.2 (-90%) and ZN.16 (reads 0 kWh; billed 11,083 kWh). A constant scale or
mapping error is invisible to any detector that compares a meter with its own history. It also puts two "known events" in doubt:
ZN.16's "switch-on" in June 2026 and ZN.8-PB.2's "new load" may be metering being corrected, not consumption starting
(the bills say ZN.16 consumed about 11 MWh a month in 2025 while the meter read zero). **Please confirm with facility staff**; this
is a hypothesis from the bills, not a finding.

**Cross-check of the alerts against an independent yardstick** (no forecast involved: at least 25% and 15 kWh away from the same
weekday's median over the previous four weeks), on 5,327 judged days in April-October 2026:

| | Alerts | Meet the yardstick | In the same direction | Lift against ordinary days |
|---|---|---|---|---|
| Ordinary days (no alert) | 5,037 | 28.7% | | 1.0x |
| Any alert | 290 | 70.3% | 86.9% | 2.4x |
| Both methods | 96 | 78.1% | 85.4% | 2.4x |
| ML model only | 58 | 81.0% | 89.7% | 2.6x |
| Simple rule only | 136 | 60.3% | 86.8% | 2.0x |

So the alerts do point at days that are unusual on a separate measure, about two and a half times as often as an ordinary day. The
yardstick is rough: of 685 independent "big deviation" days only 16% were alerted, but for 94% of the missed ones the morning
forecast also expected the change (holidays, season), so they are not anomalies by design.

**Bill actuals.** Completed 2026 bill cycles use gold-style booking, which mis-books an outage that straddles a cycle boundary.
Over 279 completed meter-cycles it differs from time-spread booking by 2% or more in only 2 (both ZN.13-PB.1), and the facility
total is identical, so no change was made. It is recorded here as a known edge case.

## 7. Defects found and fixed during validation

| # | Defect | Fix |
|---|---|---|
| 1 | Forecasts silently came back empty (NaN) beyond 92 days ahead and crashed the bill step | `MAX_HORIZON_DAYS` raised to 180; the model now raises `Errors.Misc.HorizonExceeded` instead of returning NaN |
| 2 | "The pipeline can never write to the database" was only a session default that a session could switch off itself (`SET default_transaction_read_only = off`) | `database/session.py` now refuses every statement except `SELECT * FROM api.<function>(...)`; verified against SET, `set_config`, DDL, DML, stacked statements and direct table reads |
| 3 | The `page` parameter accepted any integer, however large | bounded to 1,000,000 |
| 4 | The anomaly evaluation crashed when the validation or final-test period had no data | the ML cut-off is switched off with a clear flag, the synthetic check is skipped, and an empty development period raises `InsufficientData` |
| 5 | (process) Two of my own stress tests were vacuous: the corruption never reached the model because the chosen meter's last days were untrusted | rewritten to pick fully trusted meters and to assert that the model actually moved |
| 6 | A circular import between `utilities` and `models`: importing `models.anomaly.shift` first in a fresh interpreter failed. It only worked because the app imports modules in a lucky order | the model-dependent evaluation moved into `models/anomaly/evaluation.py`; a new test imports every module on its own in a fresh interpreter |
| 7 | A permanent doubling of one meter was reported as 9 separate level-change episodes (found by an adversarial test on the new detector) | fixed, then the whole detector was dropped for failing on real data (section 6.2) |

## 8. Open weaknesses (measured, not fixed)

1. **30-day forecasts run about 9% low overall** (forecast/actual 0.913), worst for issues in April-May (0.88 and 0.83). Whole-month accuracy per meter is 82.7% on average.
2. **A short sensor spike persists across the horizon.** A 10x error on a meter's last 3 days lifts its forecast **8x tomorrow and still 3.8x at the end of the 90 days**, because the model re-anchors on the last 7 days. Anchoring on a robust level (median or winsorised) would fix this.
3. **Losing the latest week of data moves a meter's forecast by up to -49%** (ZN.10-PB.1: ratio 0.51; others 0.93 and 1.18), since the model falls back to older data. A +20 C heatwave in the last week lowers the facility forecast by 20%.
4. **Weak meters.** ZN.16 scores 53% at 30 days: it sat at 0 kWh for months and switched on in June, and a ratio-to-recent-level model cannot forecast a switch-on. 5 of 31 meters are below 80% at 30 days.
5. **Bill forecast uses stand-in breaker ratings** (`silver.dim_meter_cb_rating` was not recovered). Only the SAR amounts depend on it, not the kWh forecasts.
6. **Constant metering or mapping errors are invisible** to every detector here (six meters disagree with the utility bills, section 6.3), and the real bills cast doubt on two of the three known events.
7. The comparison of our consumption rule with `gold.energy_consumption_validated` cannot run here (that table is not in the local database); the rule is instead validated by the exact accounting identity above.

## 9. Test inventory

| File | Tests | What it covers |
|---|---|---|
| `test_forecasting.py` | 14 | leakage, training cut-off, calendar (derived), bill rules, horizon guard |
| `test_architecture.py` | 15 | layers only depend downward, no inline SQL in repositories, explicit `__init__` exports, every endpoint marked, functions at most 30 lines |
| `test_adversarial_ingestion.py` | 18 | glitches, resets, jumps, outages, Excel matching and corrupt files |
| `test_adversarial_model.py` | 11 | synthetic edge cases; refits on perturbed real data (slow) |
| `test_validation_outputs.py` | 17 | integrity, accounting identity, forecast sanity, independent accuracy, baselines, bills, dashboard |
| `test_adversarial_anomaly.py` | 18 | fault injection, null injection, noise, starvation |
| `test_adversarial_api.py` | 34 | hostile parameters, traversal, paging, wrong methods, API-key near misses, concurrency |
| `test_adversarial_database.py` | 20 | read-only enforcement, SQL injection, UDF vs raw reconciliation |
| `test_real_bills.py` | 4 | energy accounting and data-problem detector against 28 real utility bills (needs the database) |
| `test_adversarial_alert_types.py` | 9 | incidents and night base load on synthetic data with known answers |
| `test_anomaly.py`, `test_api.py` | 15 | detector behaviour and routing |

Run `python -m pytest -q -m "not slow"` for the fast subset (about 50 seconds). The database tests need `SADINHUB_PASSWORD`.
