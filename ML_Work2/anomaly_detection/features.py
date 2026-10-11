"""Features for each checkable meter-day, using only days before it.

For a meter-day with actual A and expected E (the forecast made that morning):

  difference        e   = A - E                                    (kWh)
  size of normal    M28 = median actual kWh over the previous 28 days (trusted days only)
  denominator       D   = max(E, 0.25 x M28, 5 kWh)    -> no huge % when E is near zero
  relative          rel = e / D
  usual error       U   = 1.4826 x median |A - E| over this meter's last 28 checked
                          days (at least 14 needed), never below max(2 kWh, 5% of M28)
  standardised      z   = e / U        "how many usual errors away"
  jump              (A - previous trusted day) / max(previous, 0.25 x M28, 5)
  similar days      (A - median of the same weekday in the last 4 weeks) / max(that, 0.25 x M28, 5)
  recent level      (A - M28) / max(M28, 5)

A day's own reading never enters its own history, so nothing from the day
being checked (or later) is used to judge it.
"""
import numpy as np
import pandas as pd

RESIDUAL_WINDOW = 28        # checked days used for the usual error
MIN_RESIDUALS = 14          # minimum history before a meter-day can be scored
LEVEL_DAYS = 28             # calendar days for the recent level
MIN_LEVEL_DAYS = 7
FLOOR_KWH = 5.0             # smallest denominator, kWh
MIN_USUAL_ERROR_KWH = 2.0
MIN_USUAL_ERROR_SHARE = 0.05

FEATURES = ["z", "rel", "jump", "similar", "level"]
CLIP = {"z": 20.0, "rel": 10.0, "jump": 10.0, "similar": 10.0, "level": 10.0}


def trusted_history(daily):
    """Gold kWh on trusted days for every meter and date since the data begins."""
    trusted = daily[daily["measurable_day"]]
    return {m: g.set_index("date")["gold_kwh"].sort_index() for m, g in trusted.groupby("meter_label")}


def _features_for(actual, expected, date, history, residuals):
    """Feature values for one meter-day given that meter's earlier history."""
    recent = history[(history.index < date) & (history.index >= date - pd.Timedelta(days=LEVEL_DAYS))]
    if len(recent) < MIN_LEVEL_DAYS:
        recent = history[history.index < date].tail(MIN_LEVEL_DAYS)
    level = float(np.median(recent)) if len(recent) else np.nan
    floor = max(FLOOR_KWH, 0.25 * level) if np.isfinite(level) else FLOOR_KWH
    denom = max(expected, floor)
    diff = actual - expected

    past_res = residuals[-RESIDUAL_WINDOW:]
    if len(past_res) >= MIN_RESIDUALS and np.isfinite(level):
        usual = max(1.4826 * float(np.median(np.abs(past_res))), MIN_USUAL_ERROR_KWH, MIN_USUAL_ERROR_SHARE * level)
        history_ok = True
    else:
        usual, history_ok = np.nan, False

    before = history[(history.index < date) & (history.index >= date - pd.Timedelta(days=7))]
    prev = float(before.iloc[-1]) if len(before) else np.nan
    prev_date = before.index[-1] if len(before) else pd.NaT   # may be earlier than yesterday
    same_day = recent[recent.index.dayofweek == date.dayofweek]
    similar_ref = float(np.median(same_day)) if len(same_day) >= 2 else level

    def scaled(ref):
        return (actual - ref) / max(ref, floor) if np.isfinite(ref) else 0.0

    return {
        "diff_kwh": diff, "denominator_kwh": denom, "recent_level_kwh": level, "usual_error_kwh": usual,
        "history_ok": history_ok, "floor_kwh": floor, "previous_kwh": prev, "previous_date": prev_date,
        "similar_ref_kwh": similar_ref,
        "z": diff / usual if history_ok else np.nan,
        "rel": diff / denom,
        "jump": scaled(prev),
        "similar": scaled(similar_ref),
        "level": (actual - level) / max(level, FLOOR_KWH) if np.isfinite(level) else 0.0,
        "pct_of_expected": diff / expected if expected >= floor else np.nan,   # shown only when meaningful
    }


def build_features(frame, daily):
    """Add feature columns to every eligible row of `frame` (meter_days output)."""
    history = trusted_history(daily)
    rows = []
    for meter, part in frame.groupby("meter_label", sort=False):
        hist = history.get(meter, pd.Series(dtype=float))
        residuals = []
        for row in part.itertuples():
            out = {"index": row.Index}
            if row.eligible:
                out.update(_features_for(row.actual_kwh, row.expected_kwh, row.date, hist, residuals))
                residuals.append(row.actual_kwh - row.expected_kwh)   # becomes history only after scoring
            rows.append(out)
    feats = pd.DataFrame(rows).set_index("index")
    out = frame.join(feats)
    out["history_ok"] = out["history_ok"].astype("boolean").fillna(False).astype(bool)
    return out


def with_actual(rows, new_actual):
    """Recompute every feature for a different actual kWh, keeping the same history.

    Used for the synthetic check: only the day's own reading changes; all
    history-based references (usual error, previous day, recent level) stay.
    """
    out = rows.copy()
    a = np.asarray(new_actual, dtype=float)
    out["actual_kwh"] = a
    out["diff_kwh"] = a - out["expected_kwh"]
    out["rel"] = out["diff_kwh"] / out["denominator_kwh"]
    out["z"] = out["diff_kwh"] / out["usual_error_kwh"]

    def scaled(ref):
        ref = out[ref]
        return np.where(np.isfinite(ref), (a - ref) / np.maximum(ref, out["floor_kwh"]), 0.0)

    out["jump"] = scaled("previous_kwh")
    out["similar"] = scaled("similar_ref_kwh")
    out["level"] = (a - out["recent_level_kwh"]) / np.maximum(out["recent_level_kwh"], FLOOR_KWH)
    return out


def feature_matrix(frame):
    """Clipped feature matrix for the Isolation Forest (rows with history only)."""
    x = frame[FEATURES].astype(float).copy()
    for col, limit in CLIP.items():
        x[col] = x[col].clip(-limit, limit)
    return x.to_numpy()
