"""Names and folders of every saved file. File names carry no year, so a new year never needs a rename."""
from typing import Dict, Tuple

# key -> (folder under outputs/, file name)
ARTIFACTS: Dict[str, Tuple[str, str]] = {
    "raw_readings_database": ("data", "raw_readings_database.parquet"),
    "raw_readings_excel": ("data", "raw_readings_excel.parquet"),
    "excel_identity_audit": ("data", "excel_meter_identity.json"),
    "meters": ("data", "meters.csv"),
    "meter_breakers": ("data", "meter_breakers.csv"),
    "weather": ("data", "weather_riyadh_daily.csv"),
    "daily_actuals": ("data", "daily_actuals.parquet"),
    "validated_intervals": ("data", "validated_intervals.parquet"),
    "every_morning_forecast": ("validation", "every_morning_forecast.parquet"),
    "every_morning_forecast_extended": ("validation", "every_morning_forecast_extended.parquet"),
    "validation_detail": ("validation", "validation_detail.csv"),
    "accuracy_summary": ("validation", "accuracy_summary.csv"),
    "accuracy_summary_json": ("validation", "accuracy_summary.json"),
    "periods_not_measurable": ("validation", "periods_not_measurable_data_gaps.csv"),
    "learned_model_weights": ("validation", "learned_model_weights_by_month.csv"),
    "day_ahead_vs_actual": ("validation", "day_ahead_forecast_vs_actual.csv"),
    "long_range_accuracy": ("validation", "long_range_accuracy_facility.csv"),
    "forecast_meters": ("forecasts", "forecast_daily_meters.csv"),
    "forecast_zones": ("forecasts", "forecast_daily_zones.csv"),
    "forecast_facility": ("forecasts", "forecast_daily_facility.csv"),
    "monthly_outlook": ("forecasts", "monthly_outlook_actual_plus_forecast.csv"),
    "forecast_weights": ("forecasts", "model_weights_used_for_forecast.csv"),
    "forecast_info": ("forecasts", "forecast_info.json"),
    "bill_forecast": ("forecasts", "bill_forecast_by_meter.csv"),
    "dashboard": ("../dashboard", "SADIN_Energy_Forecast_Dashboard.html"),
    "alerts_scored": ("alerts", "every_meter_day_scored.csv"),
    "alerts_consumption": ("alerts", "consumption_alerts.csv"),
    "alerts_data_problems": ("alerts", "data_problem_alerts.csv"),
    "alerts_incidents": ("alerts", "incidents.csv"),
    "alerts_night": ("alerts", "night_load_alerts.csv"),
    "alerts_evaluation": ("alerts", "evaluation.json"),
    "alerts_dashboard": ("alerts", "alerts_dashboard.json"),
}

# Month-by-unit accuracy tables are written as accuracy_by_month_<level>_<horizon>.csv.
ACCURACY_MONTH_FOLDER: str = "validation"
