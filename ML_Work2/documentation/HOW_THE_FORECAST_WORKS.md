# SADIN Facility – how the kWh forecast works and how accurate it is

*Forecast issued 3 October 2026, using actual readings up to 2 October 2026.*

## 1. In one paragraph

Every morning the system forecasts the kWh of each of the 31 meters for every
day ahead, using only data up to yesterday. Zone and facility forecasts are the
sum of their meters, and 7-day, 30-day and monthly forecasts are the sum of the
daily ones. To prove the accuracy, every morning of 2026 (1 January to 2
October) was replayed exactly as it would have run: the model saw only the
data available that morning, and each forecast was then compared with the
validated actual kWh. **Accuracy = 100 − the average % error** (100 − WAPE).

## 2. The data and how it was checked

| Source | Period | Use |
|---|---|---|
| PostgreSQL gold layer | 9 Dec 2024 – 11 Jul 2026 | all 31 meters' readings |
| Telemetry Excel exports (31 files) | 12 Jul – 3 Oct 2026 | readings after the database |
| Open-Meteo (free weather archive, Riyadh) | 2010 – today | seasonal temperature normals |

* **Consumption is computed exactly like the gold layer.** We re-applied the
  gold rule (cumulative register minus previous valid reading, single-reading
  glitches ignored, gaps of 15+ minutes kept) to the raw database readings and
  got **identical numbers for all 14,753 meter-days**, total **2,020,697.5 kWh**,
  the same as `gold.energy_consumption_validated` and Power BI. The same rule
  was then applied to the Excel period and to Zone 13.
* **Excel files were matched to meters by the register itself**: each file's
  first kWh reading continues its meter's last database reading. This also
  identified two files with misleading names (`ZN.13-PB.1 child_2` is ZN.13-PB.2;
  `ZN.3-PB` is ZN.3-PB.1).
* **Zone 13** (not in the gold validated table) is computed from `gold.energy_raw`
  with the same rule, plus one physical check: on 5 Feb 2025 the ZN.13-PB.1
  register jumped by 2,357 kWh in 5 minutes (26,900 kW – impossible; the whole
  facility peaks at ~482 kW). That jump is not counted as consumption.
* **Missing data.** When a meter stops reporting, the gold layer books all the
  energy of the gap on the day it comes back (e.g. 79,195 kWh on one day for
  ZN.17 after a 201-day gap). Such days are not used to score single-day
  accuracy (they are not the energy of that day), but 7-day, 30-day and monthly
  totals still include them, because the register keeps counting during a gap.
  A period is left out only if an outage of over an hour spans its first or last
  midnight. The full list is in `outputs/validation/periods_not_measurable_data_gaps.csv`.

## 3. How the model works

The forecast for a meter on a future day is

> **recent level of the meter** (its last 7 days) × **expected change**

and the expected change is learned from the data:

| Driver | What it captures | Learned weight* |
|---|---|---|
| Seasonal temperature | each meter's own curve of kWh vs temperature, moved from the recent week to the **normal** temperature of the target day (season, not a weather forecast) | 0.45 (tomorrow) – 0.79 (next month) |
| Weather anomaly | the recent week was hotter/colder than normal, and that fades | 0.02 – 0.46 |
| Weekday | the meter's own weekday pattern (Fri–Sat weekend) | 0.15 – 0.41 |
| Ramadan, Eid, national days | Saudi calendar effects | −0.02 to −0.22 |
| Momentum | last 7 days vs last 28 days | −0.06 to −0.33 |
| Yesterday, last 3 days | for the next few days only | 0.80 for tomorrow |

\*Weights of the model used for the current forecast, from
`outputs/forecasts/model_weights_used_for_forecast.csv`.

* **One model shared by all 31 meters**, so rare events (Eid, the first heat of
  summer) are learned from all meters together, while each meter keeps its own
  level, temperature curve and weekday pattern.
* For **0–6 days ahead** a small gradient-boosting model (LightGBM) also learns
  each meter's short-term behaviour. For 7+ days ahead it made results worse in
  testing, so it is not used there.
* **Self-correcting:** every morning the forecast re-anchors on each meter's
  latest actual consumption, so a meter that changes behaviour is followed
  within days; every month the weights are re-learned from all data so far.
* **Nothing is hard-coded.** No accuracy figure, meter or month is entered
  anywhere in the code; the only fixed inputs are business rules (the gold
  consumption rule, the Saudi holiday calendar) and the site coordinates.
  The automatic test `tests/test_forecasting.py` scrambles all data from the
  forecast day onward and proves the forecast inputs do not change.

## 4. Results for 2026 (1 Jan – 2 Oct, every morning replayed)

| Level | Next day | Next 7 days | Next 30 days | Whole month (forecast on the 1st) |
|---|---|---|---|---|
| **Facility** | **95.3%** | **93.9%** | **90.4%** | **90.4%** |
| **Zones (21)** | **90.8%** | **88.8%** | **86.3%** | **84.2%** |
| **Meters (31)** | **89.8%** | **87.7%** | **84.9%** | **82.8%** |
| Meters at or above 80% | 31 of 31 | 28 of 31 | 26 of 31 | 23 of 31 |
| Zones at or above 80% | 21 of 21 | 20 of 21 | 18 of 21 | 15 of 21 |

Month-by-month tables for every meter and zone are in
`outputs/validation/accuracy_by_month_*.csv` and on the dashboard.

**Compared with the previous (GPT) version on its own test** – one forecast on
the 1st of each month, share of meter-months at or above 80%:
next day 71% → **75%**, 7 days 60% → **73%**, 30 days 66% → **69%**.

**How far ahead to trust it** (facility, forecasts on the 1st since Sep 2025):
days 1–30 ahead ≈ 90%, days 31–60 ≈ 86%, days 61–90 ≈ 80%
(`outputs/validation/long_range_accuracy_facility.csv`). The November and
December numbers will sharpen as the daily updates get closer to them.

## 5. Where it is below 80%, and why

Every level is above 80% overall at 1, 7 and 30 days, and every meter is above
80% for the next-day forecast. The meters below 80% on longer horizons are the
ones whose consumption changed for reasons not in the data (average kWh/day):

| Meter | What happened in the data |
|---|---|
| ZN.16-PB | **0 kWh every day from December to May**, then switched on: 59 (Jun) → 403 (Jul) → 436 (Aug) kWh/day |
| ZN.8-PB.2 | ~75–150 kWh/day until June, then a new steady load of ~340 kWh/day from July |
| ZN.11-PB.2 | irregular: 37 (Apr) → 78 → 95 → 209 (Jul) → 188 → 83 (Sep) → 164 (Oct) |
| ZN.10-PB.2 | 20 (Jan) → 85 (May) → 186 (Jun), very uneven day to day |
| ZN.5-PB | 26 (Mar) → 58 → 127 (May): a spring rise far steeper than temperature explains |
| ZN.13-PB.1 | drops to 26 kWh/day in September from 51–65 |
| ZN.4-PB.2 | alternating months: 50, 114, 53, 115 kWh/day (Jan–Apr) |

These are occupancy and operational changes (tenants, equipment switched on
or off). No model can predict them from meter readings alone; the daily
re-anchoring follows them within days, which is why next-day accuracy stays
above 80% even for these meters. **The single best way to raise these numbers
further is information the data does not contain**: planned move-ins, vacancies
and equipment changes from facility management.

## 6. Ideas tested and rejected (to keep the model honest)

Each idea was judged on a 2025 development period (forecasts issued Aug–Dec
2025), not on 2026, and kept only if it helped there without hurting 2026.

| Idea | Result |
|---|---|
| Learning from its own recent errors (bias feedback) | worse: lags behind seasonal turns |
| Separate weights per meter | no gain |
| Perfect knowledge of future weather (diagnostic only) | +1 point at 30 days → weather is not the bottleneck |
| Last year's same weeks | no gain |
| Pull toward each meter's 12-month normal | no gain |
| Blending with a simple 7-day average | no gain |
| Weighting recent months more in each meter's temperature curve | slightly worse in both periods (e.g. 30-day meter accuracy 89.2 → 88.9% in 2025, 84.9 → 84.7% in 2026) |

One consequence worth knowing: **ZN.8-PB.2's forecast rises into December**
(about 490 kWh/day against ~330 in September). Its temperature curve comes
mostly from 2025, when the unit was nearly empty in summer and used in
winter; the large load it has carried since July 2026 has no winter history
yet. The recency weighting above would have lowered this, but it made the
validated accuracy worse overall, so the model was left unchanged. Treat this
one meter's November–December figures with extra care; the daily updates will
correct it as the actual winter readings arrive.

## 7. The dashboard

`dashboard/SADIN_Energy_Forecast_Dashboard.html` (opens offline; double-click
`Open_Dashboard.cmd`). At the top, choose a **start date in 2026**, a **zone**,
a **meter or the zone total**, and a **period (day, 7 days, 30 days)**:

* the large card shows the predicted kWh for that period;
* for dates already past, the forecast shown is the one that was made **on the
  morning of that date** (using only readings up to the day before), next to
  the validated actual kWh and its accuracy – so anyone can check any day of
  2026 themselves;
* for future dates it is the latest forecast;
* a day-by-day chart, the three periods side by side, and a table of every
  zone (or every meter in the chosen zone) with forecast, actual and accuracy,
  downloadable as CSV.

Below that: what is coming (next 7/30 days, each remaining month, 2026
outlook), the 2026 track record, the year chart, accuracy by month, and an
at-a-glance grid of all zones or meters. The page shows only data produced by
the pipeline; it does not recalculate or adjust any forecast.

### Bill forecast page (SAR)

The second tab turns the forecast kWh into each meter's electricity bill
(`forecasting/billing.py`; tariff values sit at the top of that file):

* bills run in **30-day cycles from 1 January 2026** (1–30 Jan, 31 Jan–1 Mar,
  2–31 Mar, …): 12 full cycles in 2026; cycle 13 (27 Dec – 25 Jan 2027) ends
  after the forecast, so only its first 5 days of kWh are shown;
* **each meter is billed on its own**: first 6,000 kWh of the cycle at
  **0.22 SAR/kWh**, every kWh above 6,000 at **0.32 SAR/kWh**, plus the
  **service fee of its circuit breaker** (from `silver.dim_meter_cb_rating`,
  matched by meter key: 80 A → Class I 10 SAR, 100–160 A → Class II 15 SAR,
  200–250 A → Class III 21 SAR), then **× 1.15 VAT**;
  example: 10,580 kWh on 100 A = (6,000×0.22 + 4,580×0.32 + 15) × 1.15 = 3,220.69 SAR;
* completed cycles use the forecast made on the cycle's first morning (and
  also show the bill on the actual metered kWh); the cycle in progress uses
  actual kWh so far + the latest forecast; later cycles use the latest forecast;
* zone and facility bills are the sum of their meters' bills.

Every meter-cycle bill is in `outputs/forecasts/bill_forecast_2026_by_meter.csv`.

## 8. Re-running

See `README.md`: put new Excel exports in `input_excel/` and run
`run_pipeline.py --refresh`. Everything (validation, forecast, dashboard) is
rebuilt from the data; nothing needs to be edited by hand.
