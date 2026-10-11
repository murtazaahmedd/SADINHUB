"""The forecasting model, in two stages, shared by all meters.

    log(kWh + 1) = log(base + 1)                       recent level of the meter
                 + linear(physical drivers)             stage 1
                 + tree(recent behaviour)               stage 2, 0-6 days ahead only

Stage 1 is a robust (Huber) linear model on a handful of physical drivers:
seasonal temperature, weather anomaly, the meter's weekday pattern, Ramadan,
Eid and national holidays, momentum, and for the next days yesterday and the
last 3 days. Coefficients are learned separately for each band of days ahead
and re-learned every month from all data up to that month.

Stage 2 is a small LightGBM model for 0-6 days ahead that learns from what
stage 1 leaves unexplained. For 7+ days ahead the trees made forecasts worse in
development, so they are not used.
"""
from typing import Dict, List, Optional, Sequence, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

from constants.forecasting import (CATEGORICAL, FEATURES, HUBER_ITERATIONS, HUBER_RIDGE, HUBER_TUNING,
                                   MIN_TRAIN_ORIGIN, SEGMENTS, TREE_PARAMS)
from errors import Errors
from models.forecasting.features import training_rows
from utilities.guard import require, traced

Segment = Tuple[int, int, int, bool]


@traced
def linear_drivers(rows: pd.DataFrame) -> pd.DataFrame:
    """
    #Feature: Forecasting model
    #Scenario: The physical drivers of stage 1, all known on the morning of issue
    #Complexity: Time O(n) - linear in rows

    Args:
        rows: Feature rows from origin_rows.

    Returns:
        One column per driver, aligned with ``rows``.

    Raises:
        ValueError: When rows is None.
    """
    require("linear_drivers", "rows", rows)
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


@traced
def robust_linear(x, y, iterations: int = HUBER_ITERATIONS, ridge: float = HUBER_RIDGE) -> np.ndarray:
    """
    #Feature: Forecasting model
    #Scenario: Huber-weighted least squares so a few abnormal days do not drag the coefficients
    #Complexity: Time O(iterations * (n*p^2 + p^3)) - n rows, p drivers

    No intercept (the base level is the anchor).

    Args:
        x: Design matrix (n x p).
        y: Target (n).
        iterations: Reweighting passes.
        ridge: Small ridge penalty, scaled by n.

    Returns:
        Coefficient vector (p).

    Raises:
        ValueError: When x or y is None.
    """
    require("robust_linear", "x", x)
    require("robust_linear", "y", y)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    w = np.ones(len(y))
    scale = (x ** 2).mean(0) + 1e-9
    beta = np.zeros(x.shape[1])
    for _ in range(iterations):
        xw = x * w[:, None]
        beta = np.linalg.solve(xw.T @ x + ridge * len(y) * np.diag(scale), xw.T @ y)
        resid = np.abs(y - x @ beta)
        cut = HUBER_TUNING * (np.median(resid) / 0.6745 + 1e-9)
        w = np.where(resid <= cut, 1.0, cut / np.maximum(resid, 1e-12))
    return beta


class Forecaster:
    """
    #Feature: Forecasting model
    #Scenario: Fit and predict daily kWh for every meter and day ahead
    #Complexity: Time O(bands * (fit_linear + fit_tree)) - one pair per band

    Two-stage model: Huber linear drivers plus LightGBM residual trees (0-6 days ahead).
    """

    @traced
    def __init__(self, tree_params: Optional[Dict] = None, segments: Optional[Sequence[Segment]] = None) -> None:
        """
        #Feature: Forecasting model
        #Scenario: Create an unfitted model
        #Complexity: Time O(1) - constant

        Args:
            tree_params: Overrides for the LightGBM parameters.
            segments: Overrides for the (first, end, step, use_tree) bands.

        Raises:
            ValueError: When tree_params is not a mapping.
        """
        self.tree_params = dict(TREE_PARAMS, **(tree_params or {}))
        self.segments: List[Segment] = list(segments or SEGMENTS)
        self.parts: List[Tuple] = []
        self.training_rows = 0
        self.trained_through = None

    @traced
    def _fit_band(self, panel, cutoff_idx: int, band: Segment) -> Tuple:
        """
        #Feature: Forecasting model
        #Scenario: Fit one days-ahead band: robust linear stage, then optional trees on its residual
        #Complexity: Time O(n*p^2 + trees) - rows, drivers, boosted trees

        Args:
            panel: The panel.
            cutoff_idx: Only targets before this index are trained on.
            band: (first days ahead, end days ahead, morning step, use tree).

        Returns:
            ((first, end, coefficient Series, tree or None), number of rows used).

        Raises:
            ValueError: When panel is None.
        """
        require("Forecaster._fit_band", "panel", panel)
        k_lo, k_hi, step, use_tree = band
        rows = training_rows(panel, range(MIN_TRAIN_ORIGIN, cutoff_idx, step), cutoff_idx, k_lo, k_hi)
        drivers = linear_drivers(rows)
        beta = robust_linear(drivers.to_numpy(), rows["target"].to_numpy())
        tree = None
        if use_tree:
            residual = rows["target"].to_numpy() - drivers.to_numpy() @ beta
            tree = lgb.LGBMRegressor(**self.tree_params)
            tree.fit(rows[FEATURES], residual, categorical_feature=CATEGORICAL)
        return (k_lo, k_hi, pd.Series(beta, index=drivers.columns), tree), len(rows)

    @traced
    def fit(self, panel, cutoff_idx: int) -> "Forecaster":
        """
        #Feature: Forecasting model
        #Scenario: Train on everything observed before a day index
        #Complexity: Time O(bands * fit) - one fit per band

        Args:
            panel: The panel.
            cutoff_idx: Day index; only earlier targets are used.

        Returns:
            self, fitted.

        Raises:
            ValueError: When panel or cutoff_idx is None.
        """
        require("Forecaster.fit", "panel", panel)
        require("Forecaster.fit", "cutoff_idx", cutoff_idx)
        self.parts, self.training_rows = [], 0
        for band in self.segments:
            part, n_rows = self._fit_band(panel, cutoff_idx, band)
            self.parts.append(part)
            self.training_rows += n_rows
        self.trained_through = panel.dates[cutoff_idx - 1]
        return self

    @traced
    def predict_kwh(self, rows: pd.DataFrame) -> np.ndarray:
        """
        #Feature: Forecasting model
        #Scenario: Predicted kWh for every feature row
        #Complexity: Time O(rows * (p + trees)) - linear in rows

        Args:
            rows: Feature rows from origin_rows.

        Returns:
            Predicted kWh per row (never negative).

        Raises:
            ValueError: When rows is None.
            Errors.Misc.HorizonExceeded: When a row asks for more days ahead than the trained bands.
        """
        require("Forecaster.predict_kwh", "rows", rows)
        ratio = np.full(len(rows), np.nan)
        k = rows["days_ahead"].to_numpy()
        covered = np.zeros(len(rows), dtype=bool)
        for k_lo, k_hi, beta, tree in self.parts:
            sel = (k >= k_lo) & (k < k_hi)
            covered |= sel
            if not sel.any():
                continue
            part = rows.loc[sel]
            value = linear_drivers(part).to_numpy() @ beta.to_numpy()
            if tree is not None:
                value = value + tree.predict(part[FEATURES])
            ratio[sel] = value
        if not covered.all():
            raise Errors.Misc.HorizonExceeded(
                f"[Forecaster.predict_kwh] {int((~covered).sum())} rows have days_ahead outside the trained "
                f"bands (largest requested {int(k.max())}, bands end at {max(p[1] for p in self.parts) - 1}). "
                f"Fix: raise constants.forecasting.MAX_HORIZON_DAYS to cover the forecast horizon, then retrain.")
        return np.maximum(np.expm1(rows["log_base"].to_numpy() + ratio), 0.0)

    @traced
    def coefficients(self) -> pd.DataFrame:
        """
        #Feature: Forecasting model
        #Scenario: Learned stage-1 weights per band of days ahead
        #Complexity: Time O(bands * p) - small

        Returns:
            One column per band, one row per driver.

        Raises:
            RuntimeError: When the model is not fitted.
        """
        if not self.parts:
            raise RuntimeError("[Forecaster.coefficients] Model is not fitted. Invariant violated: fit() ran first. "
                               "Received: no parts. Fix: call fit() before coefficients().")
        return pd.DataFrame({f"{lo}-{hi - 1} days ahead": beta for lo, hi, beta, _ in self.parts})
