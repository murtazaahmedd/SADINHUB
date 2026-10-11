"""The forecasting model, in two stages, shared by all 31 meters.

    log(kWh + 1) = log(base + 1)                       recent level of the meter
                 + linear(physical drivers)             stage 1
                 + tree(recent behaviour)               stage 2, 0-6 days ahead only

Stage 1 is a robust linear model on a handful of physical drivers:
  * seasonal temperature - the change implied by the meter's own temperature
    curve as the season moves from the recent week to the target day (normal
    temperatures, i.e. the season, not a weather forecast);
  * weather anomaly      - how far the recent week was hotter/cooler than normal;
  * the meter's weekday pattern, Ramadan, Eid and national holidays;
  * momentum (last 7 days vs last 28) and, for the next days, yesterday and
    the last 3 days.
Linear terms extrapolate, so a season that the training data has not shown
yet (e.g. the first autumn) is still handled sensibly.

Stage 2 is a small LightGBM model for 0-6 days ahead that learns from what
stage 1 leaves unexplained (meter identity, day-to-day dynamics). For 7+ days
ahead the trees made forecasts worse in development, so they are not used.

Coefficients are learned separately for each band of days ahead, because what
predicts tomorrow (the last few days) is not what predicts next month. They are
re-learned every month from all data up to that month.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import CATEGORICAL, FEATURES, MAX_HORIZON, training_rows

TREE_PARAMS = dict(
    objective="l1", learning_rate=0.03, n_estimators=300, num_leaves=15,
    min_child_samples=300, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    reg_lambda=5.0, n_jobs=-1, verbose=-1, random_state=7,
)

# (first days-ahead, end days-ahead exclusive, train on every n-th morning, use tree)
SEGMENTS = [(0, 1, 1, True), (1, 7, 1, True), (7, 30, 2, False), (30, MAX_HORIZON, 3, False)]


def linear_drivers(rows):
    """The physical drivers of stage 1 (all known on the morning of issue)."""
    near = (rows["days_ahead"].to_numpy() < 7).astype(float)
    return pd.DataFrame({
        "seasonal_temperature": rows["seasonal_effect_change"].fillna(0.0),
        "weather_anomaly": rows["weather_anomaly_effect"].fillna(0.0),
        "weekday": rows["dow_ratio"].fillna(0.0),
        "ramadan_shift": rows["ramadan"] - rows["base_ramadan_share"],
        "holiday": (rows["eid"] | rows["national_holiday"]).astype(float),
        "base_holiday_share": rows["base_off_share"],
        "momentum": rows["trend_7_vs_28"].fillna(0.0),
        "yesterday": rows["yesterday_vs_base"].fillna(0.0).clip(-2, 2) * near,
        "last_3_days": rows["last_3_days_vs_base"].fillna(0.0).clip(-2, 2) * near,
    }, index=rows.index)


def robust_linear(x, y, iterations=15, ridge=1e-3):
    """Huber-weighted least squares, no intercept (the base level is the anchor).

    Huber weights stop a few abnormal days (a tenant moving in, a fault) from
    dragging the coefficients.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    w = np.ones(len(y))
    scale = (x ** 2).mean(0) + 1e-9
    beta = np.zeros(x.shape[1])
    for _ in range(iterations):
        xw = x * w[:, None]
        beta = np.linalg.solve(xw.T @ x + ridge * len(y) * np.diag(scale), xw.T @ y)
        resid = np.abs(y - x @ beta)
        cut = 1.345 * (np.median(resid) / 0.6745 + 1e-9)
        w = np.where(resid <= cut, 1.0, cut / np.maximum(resid, 1e-12))
    return beta


class Forecaster:
    def __init__(self, tree_params=None, segments=None):
        self.tree_params = dict(TREE_PARAMS, **(tree_params or {}))
        self.segments = segments or SEGMENTS
        self.parts = []
        self.training_rows = 0
        self.trained_through = None

    def fit(self, panel, cutoff_idx):
        """Train on everything observed before day index ``cutoff_idx``."""
        self.parts = []
        self.training_rows = 0
        for k_lo, k_hi, step, use_tree in self.segments:
            rows = training_rows(panel, range(8, cutoff_idx, step), cutoff_idx, k_lo, k_hi)
            drivers = linear_drivers(rows)
            beta = robust_linear(drivers.to_numpy(), rows["target"].to_numpy())
            tree = None
            if use_tree:
                residual = rows["target"].to_numpy() - drivers.to_numpy() @ beta
                tree = lgb.LGBMRegressor(**self.tree_params)
                tree.fit(rows[FEATURES], residual, categorical_feature=CATEGORICAL)
            self.parts.append((k_lo, k_hi, pd.Series(beta, index=drivers.columns), tree))
            self.training_rows += len(rows)
        self.trained_through = panel.dates[cutoff_idx - 1]
        return self

    def predict_kwh(self, rows):
        ratio = np.full(len(rows), np.nan)
        k = rows["days_ahead"].to_numpy()
        for k_lo, k_hi, beta, tree in self.parts:
            sel = (k >= k_lo) & (k < k_hi)
            if not sel.any():
                continue
            part = rows.loc[sel]
            value = linear_drivers(part).to_numpy() @ beta.to_numpy()
            if tree is not None:
                value = value + tree.predict(part[FEATURES])
            ratio[sel] = value
        return np.maximum(np.expm1(rows["log_base"].to_numpy() + ratio), 0.0)

    def coefficients(self):
        """Learned stage-1 weights per band of days ahead."""
        return pd.DataFrame({f"{lo}-{hi - 1} days ahead": beta for lo, hi, beta, _ in self.parts})
