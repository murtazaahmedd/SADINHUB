"""Compute the consumption alerts and rebuild the dashboard with the Alerts tab.

Usage (from the ML_Work2 folder):
    .venv\\Scripts\\python.exe -m anomaly_detection.run

Reads the saved forecasting outputs (never writes them), writes only to
anomaly_detection/outputs/, then rebuilds the dashboard page from the last
forecasting run (same as run_pipeline.py --dashboard-only). No database
connection is used.

Run it again after every forecasting run (run_pipeline.py), so the alerts use
the latest forecasts; the Alerts tab shows a notice if they are out of date.
"""
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=RuntimeWarning)

from .data import load_inputs, meter_days  # noqa: E402
from .detectors import detect  # noqa: E402
from .features import build_features  # noqa: E402
from .quality import historical_register_check, quality_alerts  # noqa: E402
from .report import alert_table, dashboard_block, evaluate  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs"
DASHBOARD_JSON = OUT / "alerts_dashboard.json"


def compute():
    daily, _ = load_inputs()
    frame = meter_days()
    inputs = frame.attrs["inputs"]
    scored, forest, settings = detect(build_features(frame, daily))
    quality = quality_alerts(scored)
    alerts = alert_table(scored)
    evaluation = evaluate(scored, quality, forest, settings, historical_register_check(daily))
    evaluation["mapping_rows_checked"] = frame.attrs["mapping_rows_checked"]
    return scored, alerts, quality, evaluation, inputs


def save(scored, alerts, quality, evaluation, inputs):
    OUT.mkdir(parents=True, exist_ok=True)
    keep = ["meter_label", "zone", "date", "period", "eligible", "history_ok", "expected_kwh", "actual_kwh",
            "diff_kwh", "usual_error_kwh", "z", "rel", "jump", "similar", "level", "ml_score", "material",
            "rule_flag", "ml_flag", "detector"]
    scored[keep].to_csv(OUT / "every_meter_day_scored_2026.csv", index=False)
    alerts.to_csv(OUT / "consumption_alerts_2026.csv", index=False)
    quality.to_csv(OUT / "data_problem_alerts_2026.csv", index=False)
    (OUT / "evaluation.json").write_text(json.dumps(evaluation, indent=2, default=str))
    block = dashboard_block(scored, alerts, quality, evaluation, inputs)
    DASHBOARD_JSON.write_text(json.dumps(block, separators=(",", ":"), default=str))
    return block


def main():
    results = compute()
    save(*results)
    from forecasting.dashboard import build_dashboard
    from forecasting.pipeline import saved_results
    path = build_dashboard(*saved_results())
    ev = results[3]
    print("Alerts per period:")
    for p in ev["periods"]:
        rate = f"{p['either_per_week']}/week" if p["either_per_week"] is not None else f"{p['days']} days, too short for a weekly rate"
        print(f"  {p['period']:12s} rule {p['rule_alerts']:4d}  ML {p['ml_alerts']:4d}  either {p['either_alerts']:4d}"
              f"  ({rate})  data problems {p['data_problem_episodes']}")
    print("Final test agreement:", ev["final_test_agreement"])
    print(f"Dashboard: {path}")


if __name__ == "__main__":
    main()
