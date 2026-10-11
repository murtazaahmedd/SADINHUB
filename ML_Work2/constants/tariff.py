"""Commercial electricity tariff (edit here if the tariff changes)."""
from typing import List, Tuple

TIER_LIMIT_KWH: float = 6000.0
RATE_UP_TO_LIMIT: float = 0.22      # SAR per kWh, first 6,000 kWh per meter per cycle
RATE_ABOVE_LIMIT: float = 0.32      # SAR per kWh, above 6,000 kWh
VAT_RATE: float = 0.15
CYCLE_DAYS: int = 30

# (lowest amps inclusive, highest amps exclusive, class, monthly fee SAR)
SERVICE_FEE_CLASSES: List[Tuple[float, float, str, float]] = [
    (1, 100, "Class I", 10.0),
    (100, 200, "Class II", 15.0),
    (200, 300, "Class III", 21.0),
    (300, 400, "Class IV", 22.0),
    (400, 400.000001, "Class V", 25.0),          # exactly 400 A
    (400.000001, float("inf"), "Class VI", 30.0),  # more than 400 A
]
