"""Facts from the data review (documentation/HOW_THE_FORECAST_WORKS.md) that the alerts are checked against.

Only the meter, the month and the direction are recorded here; the onset and
what the detectors did are measured. Update this list when a new event is confirmed.
"""
from typing import List, Tuple

# (meter label, month YYYY-MM, direction, description)
KNOWN_EVENTS: List[Tuple[str, str, str, str]] = [
    ("ZN.16-PB_child__1_ch3pm", "2026-06", "higher", "ZN.16 switched on after months at 0 kWh"),
    ("ZN.8-PB.2_child__1_ch3pm", "2026-07", "higher", "ZN.8-PB.2 took on a large new load"),
    ("ZN.13-PB.1_child__1_ch3pm", "2026-09", "lower", "ZN.13-PB.1 fell to a much lower level"),
]

# (month YYYY-MM, description) of site-wide data outages, reported as one data problem.
KNOWN_SITE_EVENTS: List[Tuple[str, str]] = [("2026-02", "February 2026 site-wide data outage")]

# (meter label, date) of a register jump that predates the forecasts.
KNOWN_REGISTER_JUMP: Tuple[str, str] = ("ZN.13-PB.1_child__1_ch3pm", "2025-02-05")
