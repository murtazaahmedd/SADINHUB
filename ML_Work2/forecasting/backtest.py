"""Walk-forward validation: replay every morning exactly as it would have run.

For each forecast issue day T the model in use was trained only on data before
the first day of T's month, and its inputs use actuals up to T-1. Nothing that
happened on or after T can reach the forecast.
"""
import time

import pandas as pd

from .features import origin_rows
from .model import Forecaster


def month_starts(panel, first_day, last_day):
    days = pd.date_range(pd.Timestamp(first_day), pd.Timestamp(last_day), freq="MS")
    if pd.Timestamp(first_day) not in days:
        days = days.insert(0, pd.Timestamp(first_day))
    return [panel.index_of(d) for d in days]


def walk_forward(panel, first_origin, last_origin, horizons=31, make_model=Forecaster, log=print):
    """Daily-issued forecasts for every origin in [first_origin, last_origin].

    Returns the forecasts and the learned coefficients of every monthly model.
    """
    first_idx, last_idx = panel.index_of(first_origin), panel.index_of(last_origin)
    starts = month_starts(panel, first_origin, last_origin)
    out, coefficients = [], {}
    for i, cutoff in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else last_idx + 1
        tick = time.time()
        model = make_model().fit(panel, cutoff)
        coefficients[panel.dates[cutoff].strftime("%Y-%m")] = model.coefficients()
        for t in range(max(cutoff, first_idx), min(end, last_idx + 1)):
            rows = origin_rows(panel, t, horizons=horizons)
            rows["pred_kwh"] = model.predict_kwh(rows)
            out.append(rows[["meter_idx", "origin_idx", "target_idx", "days_ahead", "pred_kwh", "base_kwh"]])
        log(f"  {panel.dates[cutoff]:%b %Y}: model trained on data to "
            f"{model.trained_through:%Y-%m-%d} ({model.training_rows:,} examples, {time.time() - tick:.0f}s)")
    return pd.concat(out, ignore_index=True), coefficients
