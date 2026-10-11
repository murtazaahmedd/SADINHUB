# How the consumption alerts work

*A plain-language guide to the **Alerts** tab of the SADIN dashboard, for you and for management.*

## 1. The idea in one paragraph

Every morning the forecasting model predicts each meter's kWh for that day (the
first tab of the dashboard). At the end of the day the meter's real kWh is known.
The alerts compare the two: when a meter used **clearly more or clearly less than
expected, in a way that is unusual for that particular meter**, it goes on a short
list so someone can check why. Days whose meter readings are missing or broken are
listed separately as **data problems** – they are never treated as "0 kWh used".

| | |
|---|---|
| **The forecast answers** | "How much will this meter use today?" |
| **The alerts answer** | "Now that the day is over, was it unusual?" |

## 2. What happens for every meter, every day

1. **Are the readings complete?** If the meter sent nothing, had energy from an
   outage booked on that day, had gaps of more than an hour, only reported every
   15–60 minutes, or its register jumped, the day is a **data problem** (grey) and
   is not judged. Missing time is counted only from real gaps of more than an hour;
   one continuous outage is always shown as one item, and an outage that hit many
   meters at once is shown once as a site-wide item.
2. **Compare** the actual kWh with the forecast made that morning.
3. **Is the difference unusual for this meter?** Every meter has its own "usual
   daily difference" (learned from its last 4 weeks of checked days; it needs at
   least 2 weeks first). Two methods judge this, separately:
   * **Simple rule:** the difference is at least **3 times** the meter's usual.
   * **ML model (Isolation Forest):** looks at several signals together – the
     difference from the forecast, the jump from the day before, how far the day
     is from the same weekday in recent weeks and from the meter's recent level –
     and flags combinations it rarely saw on normal days. It does **not** say which
     signal made a day unusual; the note shown with an ML-only alert is just the
     largest of the simple comparisons, for reference, not the model's own reason.
4. **Is it big enough to matter?** At least **10 kWh** and at least **20%** of what
   was expected.
5. If 1–4 pass, the day becomes an **alert** – red ▲ (more) or amber ▼ (less) –
   labelled with the method that found it: *Both methods*, *Simple rule only*, or
   *ML model only*.

Nothing from the day being judged, or from later days, is used to judge it
(an automatic test proves this).

## 3. How it was built and tested honestly

All the data was used strictly in time order:

| Period | Used for |
|---|---|
| 1 Apr 2025 – 31 Mar 2026 (checked from two weeks in: each meter needs 2 weeks of history) | The ML model learned what normal days look like, across a **full seasonal cycle**. Every early morning was forecast by a model trained only on data before that month |
| 1 Apr – 31 Aug | The ML model's alert level was set so it flags a **fixed 3%** of the days checked (about 6 alerts a week across the 31 meters, what a reviewer can work through). Five months, including a noisy spring and a calm summer, so the level is not set on an unusually quiet window. This is a staffing choice, not tied to the simple rule or to the test month |
| 1 – 30 Sep | **Final test** – untouched until the design was fixed |
| 1 – 2 Oct | Newest days, scored the same way |

There is **no list of confirmed faults**, so no "accuracy %" is claimed. The tab
shows instead:

* **alerts per week** in each period (counted from the first day a meter could be
  checked; a period of under a week gets no weekly rate),
* how often the two methods **agree** in the September test,
* **known events** from the data review – the first alert in the event's direction
  and what each method flagged in that event's first week (for example ZN.8-PB.2's
  new load was flagged by both methods on its first day, 27 June),
* a **sensitivity check** with simulated data: each checked September day is changed
  on its own by ±60%; only flags in the matching direction count. It shows the
  system reacts to a sudden one-day change – nothing more.

The exact figures are in the tab's **How well does it work?** section and in
`outputs/alerts/evaluation.json`.

## 3b. Beyond single days: incidents and night load

**Grouped incidents.** The Alerts tab opens in *grouped incidents* view. Consecutive alert days of one meter in the same
direction (with at most one quiet day between them) become **one incident**, with the total kWh difference, how many days it
lasted, whether it is **still running** at the end of the data, and an estimated money impact (the kWh difference at the lower
tariff rate, so a conservative figure, not a bill). The list is ranked by that impact, so the big events (for example ZN.16
switching on in June, about 1,600 kWh) come first. Switch to *each day separately* to see the individual days. Most alerts
are isolated days, so grouping removes only about 15% of the items; the main value is the ranking.

**Night load.** A daily total can hide a small load left running overnight. Each night (midnight to 5 am Saudi time) the average
kW is compared with the same meter's own recent nights; a night **at least 5 usual variations and at least 1 kW above its
usual level** is listed as *Night load*. In a test that added 2-10 kW to the night hours of real days, the daily detectors
caught 11-47% and the night detector 44-72%, so it finds faults the daily detectors miss. It adds about 5 alerts a week.

A third idea, flagging *permanent level changes* with a CUSUM on each meter's share of the facility, was built and tested but
**not shipped**: on real data it either flagged hundreds of episodes or never closed them, and a simpler month-on-month ratio
missed the known events. It is recorded in `documentation/VALIDATION_REPORT.md`.

**A caution about the "known events".** The real utility bills of May 2025 show ZN.16 billed for about 11 MWh while its
meter read zero, and ZN.8-PB.2 billed for 8.2 MWh while its meter read 0.7 MWh. So the June 2026 "switch-on" and "new load"
alerts may be the meters starting to report correctly rather than real consumption starting; please check with facility staff.
A meter that is wrong by a constant factor can never be flagged by comparing it with its own history.

## 4. How it helps the company

* **A short daily list** instead of checking 31 meters by hand.
* **High alerts** can point to equipment left running, longer operating hours, or a
  new tenant – possible savings once the cause is confirmed and fixed.
* **Low alerts** can point to closures, vacancies, switched-off equipment or meter faults.
* **Data problems** show meters or gateways that stopped reporting, so they can be
  fixed early and reports and bills stay correct.

## 5. What an alert is NOT

* Not proof of a fault – someone checks the cause.
* Not money saved – savings count only after action is taken.
* A permanent change (for example a new tenant) appears for the first days; then
  the forecast learns the new level and it becomes the new normal. The alerts find
  **sudden changes**, not long-running waste.

## 6. Files and how to run it

* Run after each forecasting run: `python run_pipeline.py --alerts` (or POST /alerts/run)
  (reads the saved forecasts, writes only to `outputs/alerts/`, then
  rebuilds the dashboard page; no database connection). If it has not been re-run,
  the Alerts tab shows a notice that the alerts are out of date.
* Tests: `python -m pytest tests/test_anomaly.py -q`
* `outputs/alerts/consumption_alerts.csv` – every high/low alert
* `outputs/alerts/data_problem_alerts.csv` – every data-problem period
* `outputs/alerts/every_meter_day_scored.csv` – every meter-day with its features and scores
* `outputs/alerts/evaluation.json` – all test figures
* `_backup_before_alerts_tab/` – the dashboard files exactly as they were before the tab was added
* Settings (thresholds) are at the top of `detectors.py` and `features.py`.
