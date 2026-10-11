"""Night base-load detector: equipment left running or a changed schedule that a daily total can hide.

The daily detectors judge a day's total against that morning's forecast, so a night-time load of a few kW
(10-50 kWh a day) often stays below their materiality gate. This detector compares each night's average kW
with the same meter's own recent nights, on a log scale (night loads swing with the weather).
"""
import numpy as np
import pandas as pd

from constants.anomaly import (NIGHT_LOG_OFFSET_KW, NIGHT_MIN_HISTORY, NIGHT_MIN_RISE_KW, NIGHT_SPREAD_FLOOR,
                               NIGHT_WINDOW_DAYS, NIGHT_Z, MAD_SCALE)
from utilities.guard import require, traced


@traced
def night_scores(night: pd.DataFrame) -> pd.DataFrame:
    """
    #Feature: Night base load
    #Scenario: How far above its own recent nights each night is, in the meter's usual variations
    #Complexity: Time O(n) - rolling windows per meter

    A night is judged only against earlier nights, so nothing from the night itself or later is used.

    Args:
        night: Output of night_base_load.

    Returns:
        Columns meter_label, date, night_kw, base_kw, rise_kw, spread (log units) and night_z; only nights
        with enough history.

    Raises:
        ValueError: When night is None.
    """
    require("night_scores", "night", night)
    parts = []
    for meter, d in night.sort_values("date").groupby("meter_label"):
        s = np.log(d.set_index("date")["night_kw"] + NIGHT_LOG_OFFSET_KW)
        prev = s.shift(1)
        base = prev.rolling(NIGHT_WINDOW_DAYS, min_periods=NIGHT_MIN_HISTORY).median()
        mad = (prev - base).abs().rolling(NIGHT_WINDOW_DAYS, min_periods=NIGHT_MIN_HISTORY).median() * MAD_SCALE
        spread = np.maximum(mad, NIGHT_SPREAD_FLOOR)
        parts.append(pd.DataFrame({"meter_label": meter, "date": s.index, "night_kw": np.exp(s.values) - NIGHT_LOG_OFFSET_KW,
                                   "base_kw": np.exp(base.values) - NIGHT_LOG_OFFSET_KW, "spread": spread.values,
                                   "night_z": ((s - base) / spread).values}))
    out = pd.concat(parts, ignore_index=True).dropna()
    out["rise_kw"] = out["night_kw"] - out["base_kw"]
    return out


@traced
def night_alerts(scores: pd.DataFrame) -> pd.DataFrame:
    """
    #Feature: Night base load
    #Scenario: Nights clearly above the meter's own usual night load
    #Complexity: Time O(n) - linear

    Args:
        scores: Output of night_scores.

    Returns:
        The alert rows: night_z at least 5 usual variations and at least 1 kW above the meter's usual night level.

    Raises:
        ValueError: When scores is None.
    """
    require("night_alerts", "scores", scores)
    return scores[(scores["night_z"] >= NIGHT_Z) & (scores["rise_kw"] >= NIGHT_MIN_RISE_KW)].reset_index(drop=True)
