"""Permissions follow the resource.action format. Use the module-level instances at call sites."""


class PipelinePermissions:
    """Run the pipeline and rebuild the dashboard."""

    execute = "pipeline.execute"


class ForecastPermissions:
    """Read saved forecasts."""

    read = "forecast.read"


class AccuracyPermissions:
    """Read saved accuracy tables."""

    read = "accuracy.read"


class AlertPermissions:
    """Read saved alerts and recompute them."""

    read = "alerts.read"
    execute = "alerts.execute"


class BillPermissions:
    """Read the saved bill forecast."""

    read = "bills.read"


pipeline = PipelinePermissions()
forecast = ForecastPermissions()
accuracy = AccuracyPermissions()
alerts = AlertPermissions()
bills = BillPermissions()

ALL_PERMISSIONS = frozenset({pipeline.execute, forecast.read, accuracy.read, alerts.read, alerts.execute, bills.read})
