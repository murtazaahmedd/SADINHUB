# Forecast and monthly-validation revision

Open the root `ML_Work2/dashboard.html`. August now defaults to a retrospective forecast pipeline fitted through 31 July, with inputs available before each selected start. The original July-issued view is also selectable.

Seven- and thirty-day comparisons now use audited counter endpoints when available. A reporting gap inside a period does not discard energy that the cumulative register recorded. Boundaries use the last trusted reading at/before Saudi midnight within 15 minutes. These totals include a small boundary timing approximation; exact timestamps and exclusion reasons are in `period_actuals_audit.csv`.

## 1 August, 30 days

Register meter totals available: **29/29**. At least 80% on this selected period: **26**, compared with **7** for the preserved July forecasts on the same targets.

Register zones with measured totals: **20**; zones at least 80%: **17**.


Full per-meter results: releases/forecast_revision_20261005/RESULTS.md
Comparison CSV: releases/forecast_revision_20261005/august_01_comparison.csv
