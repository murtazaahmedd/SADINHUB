"""How strongly each meter's consumption responds to Riyadh temperature.

For every meter we fit   log(kWh + 1) = level + sum_k b_k * max(T - k, 0)
                                       + weekend / Ramadan / Eid / national-day terms
with temperature hinges at 15, 20, 25, 30 and 35 C. Each meter's curve is pulled
toward the curve of all meters together (ridge shrinkage), so a meter with
little history still gets a sensible response.

The fit is redone at every month start using only data before that month, and a
forecast issued in a month uses that month's curves - never later data.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np

from constants.forecasting import (MIN_METER_FIT_DAYS, MIN_RESPONSE_FIT_DAYS, POOLED_RIDGE, SCALE_EPSILON,
                                   SHRINK_DAYS, TEMPERATURE_HINGES)
from utilities.guard import require, traced

HINGES = np.array(TEMPERATURE_HINGES)
_CONTROLS = ("weekend", "ramadan", "eid", "national_holiday")


@traced
def hinge(temp: np.ndarray) -> np.ndarray:
    """
    #Feature: Temperature response
    #Scenario: Piecewise-linear temperature terms above each hinge point
    #Complexity: Time O(n*h) - n temperatures, h hinges

    Args:
        temp: Temperatures in degrees C.

    Returns:
        Array of shape (*temp.shape, number of hinges) with max(T - k, 0).

    Raises:
        ValueError: When temp is None.
    """
    require("hinge", "temp", temp)
    temp = np.asarray(temp, dtype=float)
    return np.maximum(temp[..., None] - HINGES, 0.0)


@traced
def effect(coefs: np.ndarray, temp: np.ndarray) -> np.ndarray:
    """
    #Feature: Temperature response
    #Scenario: Temperature effect in log-kWh for every meter
    #Complexity: Time O(n*h*m) - temperatures, hinges, meters

    Args:
        coefs: Per-meter hinge slopes, shape (meters, hinges).
        temp: Temperatures in degrees C.

    Returns:
        Array of shape (len(temp), meters).

    Raises:
        ValueError: When an argument is None.
    """
    require("effect", "coefs", coefs)
    require("effect", "temp", temp)
    return hinge(temp) @ coefs.T


@traced
def _design(panel, upto: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    #Feature: Temperature response
    #Scenario: Design matrix of hinge terms and calendar controls before a cutoff
    #Complexity: Time O(n*h) - days times hinges

    Args:
        panel: Panel with temp_obs and cal.
        upto: Number of leading days to use.

    Returns:
        (design matrix, boolean mask of days with an observed temperature).

    Raises:
        ValueError: When panel is None.
    """
    require("_design", "panel", panel)
    temp = panel.temp_obs[:upto]
    controls = np.column_stack([panel.cal[c][:upto] for c in _CONTROLS])
    return np.column_stack([hinge(temp), controls]).astype(float), ~np.isnan(temp)


@traced
def _centered_meter_data(panel, cutoff_idx: int) -> List[Optional[Tuple[np.ndarray, np.ndarray]]]:
    """
    #Feature: Temperature response
    #Scenario: Per-meter centred design and target; None when too little history
    #Complexity: Time O(m*n*h) - meters, days, hinges

    Args:
        panel: Panel with y, temp_obs and cal.
        cutoff_idx: Only days before this index are used.

    Returns:
        One (x, y) pair per meter, or None.

    Raises:
        ValueError: When panel is None.
    """
    require("_centered_meter_data", "panel", panel)
    x_all, temp_ok = _design(panel, cutoff_idx)
    y_all = np.log1p(panel.y[:, :cutoff_idx])
    per_meter: List[Optional[Tuple[np.ndarray, np.ndarray]]] = []
    for i in range(len(panel.meters)):
        ok = temp_ok & ~np.isnan(y_all[i])
        if ok.sum() < MIN_METER_FIT_DAYS:
            per_meter.append(None)
            continue
        per_meter.append((x_all[ok] - x_all[ok].mean(0), y_all[i, ok] - y_all[i, ok].mean()))
    return per_meter


@traced
def _pooled_fit(per_meter: List[Optional[Tuple[np.ndarray, np.ndarray]]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    #Feature: Temperature response
    #Scenario: One curve fitted on all meters together, used as the shrinkage target
    #Complexity: Time O(N*p^2) - N stacked rows, p regressors

    Args:
        per_meter: Output of _centered_meter_data with at least one non-None item.

    Returns:
        (pooled coefficients, per-regressor scale).

    Raises:
        ValueError: When per_meter is None.
    """
    require("_pooled_fit", "per_meter", per_meter)
    xp = np.vstack([item[0] for item in per_meter if item is not None])
    yp = np.concatenate([item[1] for item in per_meter if item is not None])
    scale = (xp ** 2).mean(0) + SCALE_EPSILON
    pooled = np.linalg.solve(xp.T @ xp + POOLED_RIDGE * np.diag(scale) * len(yp), xp.T @ yp)
    return pooled, scale


@traced
def fit(panel, cutoff_idx: int) -> np.ndarray:
    """
    #Feature: Temperature response
    #Scenario: Per-meter hinge slopes from data before a cutoff
    #Complexity: Time O(m*p^3 + N*p^2) - meters, regressors, stacked rows

    Args:
        panel: Panel with meters, y, temp_obs and cal.
        cutoff_idx: Only days before this index are used.

    Returns:
        Array (meters x hinges); zeros when fewer than 60 days exist.

    Raises:
        ValueError: When panel is None.
    """
    require("fit", "panel", panel)
    coefs = np.zeros((len(panel.meters), len(HINGES)))
    if cutoff_idx < MIN_RESPONSE_FIT_DAYS:
        return coefs
    per_meter = _centered_meter_data(panel, cutoff_idx)
    if all(item is None for item in per_meter):
        return coefs
    pooled, scale = _pooled_fit(per_meter)
    penalty = SHRINK_DAYS * np.diag(scale)
    for i, item in enumerate(per_meter):
        if item is None:
            coefs[i] = pooled[: len(HINGES)]
            continue
        x, y = item
        coefs[i] = np.linalg.solve(x.T @ x + penalty, x.T @ y + penalty @ pooled)[: len(HINGES)]
    return coefs


class ResponseBook:
    """
    #Feature: Temperature response
    #Scenario: Curves fitted at each month start, looked up by forecast issue day
    #Complexity: Time O(months * fit) - one fit per month

    Curves fitted at each month start; lookup by forecast issue day.
    """

    @traced
    def __init__(self, panel) -> None:
        """
        #Feature: Temperature response
        #Scenario: Fit one curve set per calendar month of the panel
        #Complexity: Time O(months * fit)

        Args:
            panel: Panel with dates, meters, y, temp_obs and cal.

        Raises:
            ValueError: When panel is None.
        """
        require("ResponseBook.__init__", "panel", panel)
        months = panel.dates.to_period("M")
        self.month_start_idx: Dict = {}
        for idx, (period, first) in enumerate(zip(months, np.r_[True, months[1:] != months[:-1]])):
            if first:
                self.month_start_idx[period] = idx
        self.curves = {p: fit(panel, i) for p, i in self.month_start_idx.items()}
        self.months = months

    @traced
    def coefs_for(self, origin_idx: int) -> np.ndarray:
        """
        #Feature: Temperature response
        #Scenario: Curves in force on a forecast issue day
        #Complexity: Time O(1) - constant

        Args:
            origin_idx: Day index of the forecast issue day.

        Returns:
            Per-meter hinge slopes fitted at that month's start.

        Raises:
            ValueError: When origin_idx is None.
        """
        require("ResponseBook.coefs_for", "origin_idx", origin_idx)
        return self.curves[self.months[min(origin_idx, len(self.months) - 1)]]
