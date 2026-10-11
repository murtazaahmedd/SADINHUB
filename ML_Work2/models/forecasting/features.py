"""Model inputs for every (meter, forecast issue day, days ahead).

A forecast is issued on the morning of day T using actuals up to and including
T-1 (nothing later). It covers T, T+1, ... Every feature below is computable on
the morning of T, which is checked by tests/test_forecasting.py.

The model predicts   log(kWh on target day + 1) - log(base level + 1)
where the base level is the meter's own recent consumption. Forecasting a
*ratio to the latest level* is what makes the system self-correcting: every
morning it re-anchors on what the meter actually did in the last week.
"""
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from constants.calendar import CLIP_EID_ADHA_DAYS, CLIP_RAMADAN_DAYS
from constants.forecasting import (BASE_WINDOW_DAYS, FALLBACK_BASE_DAYS, LAST_YEAR_HALF_WINDOW, LONG_WINDOW_DAYS,
                                   MAX_HORIZON_DAYS, MIN_BASE_DAYS_LONG, MIN_BASE_DAYS_SHORT, MIN_DOW_DAYS,
                                   MIN_LAST_YEAR_DAYS, MIN_RECENT3_DAYS, MIN_TRAIN_ORIGIN, RECENT_DAYS,
                                   SENSITIVITY_TEMP_C, YEAR_LAG_DAYS)
from models.forecasting.temperature_response import ResponseBook, effect, hinge
from utilities.guard import require, traced
from utilities.saudi_calendar import calendar_features
from utilities.weather_normals import climate_normal

_CAL_COLUMNS = ("dow", "weekend", "ramadan", "eid", "national_holiday", "off_day",
                "days_from_ramadan_start", "days_from_eid_adha")


class Panel:
    """
    #Feature: Forecast features
    #Scenario: Dense meter x date arrays plus cumulative sums for fast window means
    #Complexity: Time O(m*n) - meters times days

    Dense meter x date arrays plus cumulative sums for fast window means.
    """

    @traced
    def __init__(self, daily: pd.DataFrame, weather: pd.DataFrame, end_date) -> None:
        """
        #Feature: Forecast features
        #Scenario: Build every array the feature builder needs
        #Complexity: Time O(m*n) - meters times days

        Args:
            daily: Daily table (meter_label, date, spread_kwh, gold_kwh, flags, zone).
            weather: Observed daily temperature (date, t_mean, t_max).
            end_date: Last day the panel covers (the forecast end).

        Raises:
            ValueError: When an argument is None.
        """
        require("Panel.__init__", "daily", daily)
        require("Panel.__init__", "weather", weather)
        require("Panel.__init__", "end_date", end_date)
        self.meters = sorted(daily["meter_label"].unique())
        self.dates = pd.date_range(daily["date"].min(), pd.Timestamp(end_date), freq="D")
        self._fill_daily_arrays(daily)
        self._fill_calendar(self.dates)
        self._fill_temperature(weather)
        self._fill_cumulative_tables()
        self._last_meas = self._last_measurable_index()
        self.responses = ResponseBook(self)

    @traced
    def _fill_daily_arrays(self, daily: pd.DataFrame) -> None:
        """
        #Feature: Forecast features
        #Scenario: Place the daily table into meter x date arrays
        #Complexity: Time O(rows) - linear

        Args:
            daily: The daily table.

        Raises:
            ValueError: When daily is None.
        """
        require("Panel._fill_daily_arrays", "daily", daily)
        m, n = len(self.meters), len(self.dates)
        row = {lab: i for i, lab in enumerate(self.meters)}
        ri = daily["meter_label"].map(row).to_numpy()
        ci = pd.Series(np.arange(n), index=self.dates).reindex(daily["date"]).to_numpy()
        self.y = np.full((m, n), np.nan)        # trusted daily kWh (training/levels)
        self.gold = np.full((m, n), np.nan)     # gold-booked daily kWh (scoring)
        self.measurable = np.zeros((m, n), bool)
        self.outage_start = np.zeros((m, n), bool)
        self.exists = np.zeros((m, n), bool)
        meas = daily["measurable_day"].to_numpy()
        self.y[ri[meas], ci[meas]] = daily["spread_kwh"].to_numpy()[meas]
        comp = daily["complete_day"].to_numpy()
        self.gold[ri[comp], ci[comp]] = daily["gold_kwh"].to_numpy()[comp]
        self.measurable[ri, ci] = meas
        self.outage_start[ri, ci] = daily["outage_at_day_start"].to_numpy()
        self.exists[ri, ci] = True
        zones = daily.drop_duplicates("meter_label").set_index("meter_label")["zone"]
        self.zone = zones.reindex(self.meters).to_numpy().astype(int)

    @traced
    def _fill_calendar(self, dates: pd.DatetimeIndex) -> None:
        """
        #Feature: Forecast features
        #Scenario: Calendar arrays for every panel day
        #Complexity: Time O(n) - days

        Args:
            dates: Every day of the panel.

        Raises:
            ValueError: When dates is None.
        """
        require("Panel._fill_calendar", "dates", dates)
        cal = calendar_features(pd.Series(dates)).reset_index(drop=True)
        self.cal = {c: cal[c].to_numpy() for c in cal.columns}
        self.cal["days_from_ramadan_start"] = np.clip(self.cal["days_from_ramadan_start"],
                                                      -CLIP_RAMADAN_DAYS, CLIP_RAMADAN_DAYS)
        self.cal["days_from_eid_adha"] = np.clip(self.cal["days_from_eid_adha"],
                                                 -CLIP_EID_ADHA_DAYS, CLIP_EID_ADHA_DAYS)

    @traced
    def _fill_temperature(self, weather: pd.DataFrame) -> None:
        """
        #Feature: Forecast features
        #Scenario: Observed temperature and one climate normal per year (earlier years only)
        #Complexity: Time O(years * n) - years times days

        Args:
            weather: Observed daily temperature.

        Raises:
            ValueError: When weather is None.
        """
        require("Panel._fill_temperature", "weather", weather)
        self.temp_obs = weather.set_index("date")["t_mean"].reindex(self.dates).to_numpy()
        self.temp_normal: Dict[int, np.ndarray] = {}
        doy = np.clip(self.dates.dayofyear.to_numpy(), 1, 365)
        for year in sorted(set(self.dates.year)):
            normal = climate_normal(weather, before_year=year)["normal_t_mean"]
            self.temp_normal[year] = normal.reindex(doy).to_numpy()

    @traced
    def _fill_cumulative_tables(self) -> None:
        """
        #Feature: Forecast features
        #Scenario: Cumulative sums that make any window mean O(1)
        #Complexity: Time O(m*n) - meters times days

        Raises:
            RuntimeError: When the arrays have not been filled yet.
        """
        self._cs, self._cn = self._cumsum(self.y)
        self._dow_cs, self._dow_cn = {}, {}
        for w in range(7):
            masked = np.where(self.cal["dow"][None, :] == w, self.y, np.nan)
            self._dow_cs[w], self._dow_cn[w] = self._cumsum(masked)
        self._ram = np.r_[0, np.cumsum(self.cal["ramadan"])]
        self._off = np.r_[0, np.cumsum(self.cal["eid"] | self.cal["national_holiday"])]
        self._tcs = np.r_[0, np.cumsum(np.nan_to_num(self.temp_obs))]
        self._tcn = np.r_[0, np.cumsum(~np.isnan(self.temp_obs))]

    @staticmethod
    @traced
    def _cumsum(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        #Feature: Forecast features
        #Scenario: Cumulative sum and count that ignore missing values
        #Complexity: Time O(m*n) - linear

        Args:
            values: Meter x day array with NaN for missing.

        Returns:
            (cumulative sums, cumulative counts), each with a leading zero column.

        Raises:
            ValueError: When values is None.
        """
        require("Panel._cumsum", "values", values)
        ok = ~np.isnan(values)
        cs = np.concatenate([np.zeros((values.shape[0], 1)), np.cumsum(np.where(ok, values, 0), 1)], 1)
        cn = np.concatenate([np.zeros((values.shape[0], 1)), np.cumsum(ok, 1)], 1)
        return cs, cn

    @traced
    def _last_measurable_index(self) -> np.ndarray:
        """
        #Feature: Forecast features
        #Scenario: Index of each meter's most recent trusted day, at every day
        #Complexity: Time O(m*n) - linear

        Returns:
            Meter x day array of the latest trusted day index so far (-1 when none).

        Raises:
            RuntimeError: When y has not been filled.
        """
        idx = np.where(~np.isnan(self.y), np.arange(self.y.shape[1])[None, :], -1)
        return np.maximum.accumulate(idx, axis=1)

    @property
    def last_measurable(self) -> np.ndarray:
        """Meter x day array of the latest trusted day index so far."""
        return self._last_meas

    @traced
    def window_mean(self, a, b, cs: Optional[np.ndarray] = None, cn: Optional[np.ndarray] = None):
        """
        #Feature: Forecast features
        #Scenario: Mean of y over day indices [a, b) per meter
        #Complexity: Time O(m) - constant per meter

        Args:
            a: First day index (scalar or meter x horizon array).
            b: End day index, exclusive.
            cs: Optional cumulative sums to use instead of y's.
            cn: Optional cumulative counts to use instead of y's.

        Returns:
            (means with NaN where no data, counts).

        Raises:
            ValueError: When a or b is None.
        """
        require("Panel.window_mean", "a", a)
        require("Panel.window_mean", "b", b)
        cs = self._cs if cs is None else cs
        cn = self._cn if cn is None else cn
        a = np.clip(a, 0, cs.shape[1] - 1)
        b = np.clip(b, 0, cs.shape[1] - 1)
        rows = np.arange(cs.shape[0])[:, None] if np.ndim(a) == 2 else np.arange(cs.shape[0])
        total = cs[rows, b] - cs[rows, a]
        count = cn[rows, b] - cn[rows, a]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(count > 0, total / np.maximum(count, 1), np.nan), count

    @traced
    def index_of(self, day) -> int:
        """
        #Feature: Forecast features
        #Scenario: Day index of a calendar date
        #Complexity: Time O(1) - constant

        Args:
            day: Anything convertible to a Timestamp.

        Returns:
            Days since the first panel day.

        Raises:
            ValueError: When day is None.
        """
        require("Panel.index_of", "day", day)
        return int((pd.Timestamp(day) - self.dates[0]).days)


@traced
def _base_level(panel: Panel, t: int, m: int) -> Dict[str, np.ndarray]:
    """
    #Feature: Forecast features
    #Scenario: The meter's recent level that every forecast is anchored on
    #Complexity: Time O(m) - constant per meter

    Args:
        panel: The panel.
        t: Forecast issue day index.
        m: Number of meters.

    Returns:
        Dict with lvl7, lvl28, base, base_age and has_base arrays.

    Raises:
        ValueError: When panel is None.
    """
    require("_base_level", "panel", panel)
    lvl7, c7 = panel.window_mean(t - BASE_WINDOW_DAYS, t)
    lvl28, c28 = panel.window_mean(t - LONG_WINDOW_DAYS, t)
    last = panel.last_measurable[:, t - 1] if t > 0 else np.full(m, -1)
    recent_start = np.maximum(last - (FALLBACK_BASE_DAYS - 1), 0)
    lvl_last, _ = panel.window_mean(recent_start, np.maximum(last + 1, 0))
    base = np.where(c7 >= MIN_BASE_DAYS_SHORT, lvl7,
                    np.where(c28 >= MIN_BASE_DAYS_LONG, lvl28, lvl_last))
    return {"lvl7": lvl7, "lvl28": lvl28, "base": base, "base_age": t - 1 - last,
            "has_base": (last >= 0) & ~np.isnan(base)}


@traced
def _recent_behaviour(panel: Panel, t: int, m: int, log_base: np.ndarray) -> Dict[str, np.ndarray]:
    """
    #Feature: Forecast features
    #Scenario: Yesterday, two days ago, last three days and 28-day volatility versus the base
    #Complexity: Time O(m) - constant per meter

    Args:
        panel: The panel.
        t: Forecast issue day index.
        m: Number of meters.
        log_base: log(base + 1) per meter.

    Returns:
        Dict of per-meter arrays.

    Raises:
        ValueError: When panel is None.
    """
    require("_recent_behaviour", "panel", panel)
    yesterday = panel.y[:, t - 1] if t > 0 else np.full(m, np.nan)
    two_days_ago = panel.y[:, t - 2] if t > 1 else np.full(m, np.nan)
    lvl3, c3 = panel.window_mean(t - RECENT_DAYS, t)
    lvl3 = np.where(c3 >= MIN_RECENT3_DAYS, lvl3, np.nan)
    recent = np.log1p(panel.y[:, max(t - LONG_WINDOW_DAYS, 0):t])
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        volatility = np.nanstd(recent, axis=1) if recent.shape[1] else np.full(m, np.nan)
    return {"yesterday_vs_base": np.log1p(yesterday) - log_base,
            "two_days_ago_vs_base": np.log1p(two_days_ago) - log_base,
            "last_3_days_vs_base": np.log1p(lvl3) - log_base, "volatility_28": volatility}


@traced
def _weekday_pattern(panel: Panel, t: int, d: np.ndarray, dd: np.ndarray, m: int, horizons: int,
                     log_base: np.ndarray, lvl28: np.ndarray) -> Dict[str, np.ndarray]:
    """
    #Feature: Forecast features
    #Scenario: The meter's weekday pattern and the same weekday last week
    #Complexity: Time O(7*m*h) - weekdays, meters, horizons

    Args:
        panel: The panel.
        t: Forecast issue day index.
        d: Target day indices.
        dd: Target day indices clipped into the panel.
        m: Number of meters.
        horizons: Days ahead requested.
        log_base: log(base + 1) per meter.
        lvl28: 28-day mean level per meter.

    Returns:
        Dict with dow_ratio and same_day_last_week_vs_base (meters x horizons).

    Raises:
        ValueError: When panel is None.
    """
    require("_weekday_pattern", "panel", panel)
    dow_t = panel.cal["dow"][dd]
    dow_mean = np.full((m, horizons), np.nan)
    for w in range(7):
        cols = dow_t == w
        if cols.any():
            mean_w, cnt = panel.window_mean(t - LONG_WINDOW_DAYS, t, panel._dow_cs[w], panel._dow_cn[w])
            dow_mean[:, cols] = np.where(cnt >= MIN_DOW_DAYS, mean_w, np.nan)[:, None]
    n = len(panel.dates)
    lag7_idx = d - 7
    same_week = np.where(lag7_idx[None, :] < t, panel.y[:, np.clip(lag7_idx, 0, n - 1)], np.nan)
    return {"dow_ratio": np.log1p(dow_mean) - np.log1p(lvl28)[:, None],
            "same_day_last_week_vs_base": np.log1p(same_week) - log_base[:, None]}


@traced
def _last_year_change(panel: Panel, t: int, d: np.ndarray, m: int) -> Dict[str, np.ndarray]:
    """
    #Feature: Forecast features
    #Scenario: Last year's change from the base window to the target day (weekday-aligned)
    #Complexity: Time O(m*h) - meters times horizons

    Args:
        panel: The panel.
        t: Forecast issue day index.
        d: Target day indices.
        m: Number of meters.

    Returns:
        Dict with last_year_change and last_year_available (meters x horizons).

    Raises:
        ValueError: When panel is None.
    """
    require("_last_year_change", "panel", panel)
    half = LAST_YEAR_HALF_WINDOW
    centre = (d - YEAR_LAG_DAYS)[None, :].repeat(m, 0)
    ly_t, ly_tc = panel.window_mean(centre - half, centre + half + 1)
    ly_b, ly_bc = panel.window_mean(t - YEAR_LAG_DAYS - BASE_WINDOW_DAYS, t - YEAR_LAG_DAYS)
    ly_ok = ((ly_tc >= MIN_LAST_YEAR_DAYS) & (ly_bc[:, None] >= MIN_LAST_YEAR_DAYS)
             & (d[None, :] - YEAR_LAG_DAYS - half >= 0))
    change = np.where(ly_ok, np.log1p(ly_t) - np.log1p(ly_b)[:, None], np.nan)
    return {"last_year_change": change, "last_year_available": ly_ok.astype(int)}


@traced
def _temperature_scalars(panel: Panel, t: int, dd: np.ndarray) -> Dict[str, object]:
    """
    #Feature: Forecast features
    #Scenario: Normal temperature of the target days and the recent observed temperature
    #Complexity: Time O(h) - horizons

    Args:
        panel: The panel.
        t: Forecast issue day index.
        dd: Target day indices clipped into the panel.

    Returns:
        Dict with normal, normal_target, normal_base and recent_temp.

    Raises:
        ValueError: When panel is None.
    """
    require("_temperature_scalars", "panel", panel)
    n = len(panel.dates)
    start = max(t - BASE_WINDOW_DAYS, 0)
    normal = panel.temp_normal[panel.dates[min(t, n - 1)].year]
    tc = panel._tcn[t] - panel._tcn[start]
    return {"normal": normal, "normal_target": normal[dd],
            "normal_base": np.nanmean(normal[start:t]) if t > 0 else np.nan,
            "recent_temp": (panel._tcs[t] - panel._tcs[start]) / tc if tc > 0 else np.nan}


@traced
def _temperature_effects(panel: Panel, t: int, m: int, scal: Dict[str, object]) -> Dict[str, np.ndarray]:
    """
    #Feature: Forecast features
    #Scenario: Change implied by each meter's temperature curve, split into season and weather anomaly
    #Complexity: Time O(m*h*k) - meters, horizons, hinges

    Args:
        panel: The panel.
        t: Forecast issue day index.
        m: Number of meters.
        scal: Output of _temperature_scalars.

    Returns:
        Dict with temp_effect_change, seasonal_effect_change, weather_anomaly_effect, temp_sensitivity.

    Raises:
        ValueError: When panel is None.
    """
    require("_temperature_effects", "panel", panel)
    start = max(t - BASE_WINDOW_DAYS, 0)
    coefs = panel.responses.coefs_for(t)
    recent_obs = panel.temp_obs[start:t]
    recent_obs = recent_obs[~np.isnan(recent_obs)]
    recent_effect = hinge(recent_obs).mean(0) @ coefs.T if len(recent_obs) else np.full(m, np.nan)
    target_effect = effect(coefs, scal["normal_target"]).T
    base_normal = scal["normal"][start:t]
    base_normal_effect = hinge(base_normal).mean(0) @ coefs.T if len(base_normal) else np.full(m, np.nan)
    return {"temp_effect_change": target_effect - recent_effect[:, None],
            "seasonal_effect_change": target_effect - base_normal_effect[:, None],
            "weather_anomaly_effect": base_normal_effect - recent_effect,
            "temp_sensitivity": effect(coefs, np.array([SENSITIVITY_TEMP_C]))[0]}


@traced
def _broadcast_blocks(panel: Panel, dd: np.ndarray, rec: Dict[str, np.ndarray], grid) -> Dict[str, np.ndarray]:
    """
    #Feature: Forecast features
    #Scenario: Recent-behaviour and calendar columns broadcast to meters x horizons
    #Complexity: Time O(m*h) - meters times horizons

    Args:
        panel: The panel.
        dd: Target day indices clipped into the panel.
        rec: Output of _recent_behaviour.
        grid: Broadcasting helper.

    Returns:
        Column name to array.

    Raises:
        ValueError: When panel is None.
    """
    require("_broadcast_blocks", "panel", panel)
    out = {name: grid(rec[name][:, None])
           for name in ("yesterday_vs_base", "two_days_ago_vs_base", "last_3_days_vs_base", "volatility_28")}
    out.update({c: grid(panel.cal[c][dd][None, :]) for c in _CAL_COLUMNS})
    return out


@traced
def _assemble_frame(panel: Panel, t: int, m: int, d: np.ndarray, dd: np.ndarray, k: np.ndarray,
                    parts: Dict[str, Dict]) -> Dict[str, np.ndarray]:
    """
    #Feature: Forecast features
    #Scenario: Combine every feature block into broadcastable columns
    #Complexity: Time O(m*h) - meters times horizons

    Args:
        panel: The panel.
        t: Forecast issue day index.
        m: Number of meters.
        d: Target day indices.
        dd: Target day indices clipped into the panel.
        k: Days ahead.
        parts: Blocks from the feature helpers (base, recent, weekday, ly, scal, temp).

    Returns:
        Column name to array (meters x horizons or broadcastable).

    Raises:
        ValueError: When panel is None.
    """
    require("_assemble_frame", "panel", panel)
    base, rec, wk, ly = parts["base"], parts["recent"], parts["weekday"], parts["ly"]
    scal, tmp, start = parts["scal"], parts["temp"], max(t - BASE_WINDOW_DAYS, 0)

    def grid(values):
        return np.broadcast_to(values, (m, len(k)))

    log_base = np.log1p(base["base"])
    frame = {
        "meter_idx": grid(np.arange(m)[:, None]), "origin_idx": grid(np.full((1, 1), t)),
        "target_idx": grid(d[None, :]), "days_ahead": grid(k[None, :]), "meter_code": grid(np.arange(m)[:, None]),
        "base_kwh": grid(base["base"][:, None]), "log_base": grid(log_base[:, None]),
        "base_age_days": grid(base["base_age"][:, None]),
        "trend_7_vs_28": grid((np.log1p(base["lvl7"]) - np.log1p(base["lvl28"]))[:, None]),
        "base_ramadan_share": grid(np.full((1, 1), (panel._ram[t] - panel._ram[start]) / 7.0)),
        "base_off_share": grid(np.full((1, 1), (panel._off[t] - panel._off[start]) / 7.0)),
        "normal_temp_target": grid(scal["normal_target"][None, :]),
        "normal_temp_shift": grid((scal["normal_target"] - scal["normal_base"])[None, :]),
        "recent_temp_anomaly": grid(np.full((1, 1), scal["recent_temp"] - scal["normal_base"])),
        "recent_temp": grid(np.full((1, 1), scal["recent_temp"])),
        "has_base": grid(base["has_base"][:, None]),
        "last_year_change": ly["last_year_change"], "last_year_available": ly["last_year_available"],
        "dow_ratio": wk["dow_ratio"], "same_day_last_week_vs_base": wk["same_day_last_week_vs_base"],
        "temp_effect_change": tmp["temp_effect_change"], "seasonal_effect_change": tmp["seasonal_effect_change"],
        "weather_anomaly_effect": grid(tmp["weather_anomaly_effect"][:, None]),
        "temp_sensitivity": grid(tmp["temp_sensitivity"][:, None]),
    }
    frame.update(_broadcast_blocks(panel, dd, rec, grid))
    return frame


@traced
def origin_rows(panel: Panel, t: int, horizons: int = MAX_HORIZON_DAYS, meters=None) -> pd.DataFrame:
    """
    #Feature: Forecast features
    #Scenario: Feature rows for one forecast issue day, every meter and every day ahead
    #Complexity: Time O(m*h) - meters times horizons

    Args:
        panel: The panel.
        t: Forecast issue day index.
        horizons: Number of days ahead.
        meters: Optional subset of meter indices.

    Returns:
        One row per meter and day ahead, with actual_kwh and target (NaN when not yet known).

    Raises:
        ValueError: When panel or t is None.
    """
    require("origin_rows", "panel", panel)
    require("origin_rows", "t", t)
    m = len(panel.meters)
    meters = np.arange(m) if meters is None else np.asarray(meters)
    k = np.arange(horizons)
    d = t + k
    dd = np.clip(d, 0, len(panel.dates) - 1)
    base = _base_level(panel, t, m)
    log_base = np.log1p(base["base"])
    scal = _temperature_scalars(panel, t, dd)
    parts = {"base": base, "recent": _recent_behaviour(panel, t, m, log_base),
             "weekday": _weekday_pattern(panel, t, d, dd, m, horizons, log_base, base["lvl28"]),
             "ly": _last_year_change(panel, t, d, m), "scal": scal,
             "temp": _temperature_effects(panel, t, m, scal)}
    frame = _assemble_frame(panel, t, m, d, dd, k, parts)
    out = pd.DataFrame({key: np.asarray(val)[meters].ravel() for key, val in frame.items()})
    keep = (out["target_idx"].to_numpy() < len(panel.dates)) & out["has_base"].to_numpy()
    out = out[keep].drop(columns="has_base")
    out["actual_kwh"] = panel.y[out["meter_idx"].to_numpy(), out["target_idx"].to_numpy()]
    out["target"] = np.log1p(out["actual_kwh"]) - out["log_base"]
    return out.reset_index(drop=True)


@traced
def training_rows(panel: Panel, origin_indices, cutoff_idx: int, k_lo: int = 0,
                  k_hi: int = MAX_HORIZON_DAYS) -> pd.DataFrame:
    """
    #Feature: Forecast features
    #Scenario: Training rows whose targets were observed before the cutoff
    #Complexity: Time O(origins * m * h) - origins, meters, horizons

    Args:
        panel: The panel.
        origin_indices: Forecast issue day indices to use.
        cutoff_idx: Only targets before this index are used.
        k_lo: Lowest days ahead included.
        k_hi: Highest days ahead excluded.

    Returns:
        Feature rows with a non-missing target.

    Raises:
        ValueError: When panel is None.
    """
    require("training_rows", "panel", panel)
    frames = []
    for t in origin_indices:
        if t >= cutoff_idx or t < MIN_TRAIN_ORIGIN:
            continue
        rows = origin_rows(panel, t, horizons=min(k_hi, cutoff_idx - t))
        rows = rows[(rows["days_ahead"] >= k_lo) & (rows["target_idx"] < cutoff_idx)]
        frames.append(rows.dropna(subset=["target"]))
    return pd.concat(frames, ignore_index=True)
