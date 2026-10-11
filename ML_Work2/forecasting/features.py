"""Model inputs for every (meter, forecast issue day, days ahead).

A forecast is issued on the morning of day T using actuals up to and including
T-1 (nothing later). It covers T, T+1, ... T+H-1. Every feature below is
computable on the morning of T, which is checked by tests/test_no_leakage.py.

The model predicts   log(kWh on target day + 1) - log(base level + 1)
where the base level is the meter's own recent consumption. Forecasting a
*ratio to the latest level* is what makes the system self-correcting: every
morning it re-anchors on what the meter actually did in the last week.
"""
import warnings

import numpy as np
import pandas as pd

from .saudi_calendar import calendar_features
from .temperature_response import ResponseBook, effect, hinge
from .weather import climate_normal

MAX_HORIZON = 180  # long enough to cover today to 31 December (data ending 10 Jul needs 174)

FEATURES = [
    "days_ahead", "meter_code", "log_base", "base_age_days", "trend_7_vs_28", "yesterday_vs_base",
    "two_days_ago_vs_base", "last_3_days_vs_base", "volatility_28",
    "dow_ratio", "same_day_last_week_vs_base", "last_year_change", "last_year_available",
    "dow", "weekend", "ramadan", "eid", "national_holiday", "off_day",
    "days_from_ramadan_start", "days_from_eid_adha",
    "base_ramadan_share", "base_off_share",
    "normal_temp_target", "normal_temp_shift", "recent_temp_anomaly", "recent_temp",
    "temp_effect_change", "seasonal_effect_change", "weather_anomaly_effect", "temp_sensitivity",
]
CATEGORICAL = ["meter_code"]


class Panel:
    """Dense meter x date arrays plus cumulative sums for fast window means."""

    def __init__(self, daily, weather, end_date):
        self.meters = sorted(daily["meter_label"].unique())
        start = daily["date"].min()
        self.dates = pd.date_range(start, pd.Timestamp(end_date), freq="D")
        m, n = len(self.meters), len(self.dates)
        row = {lab: i for i, lab in enumerate(self.meters)}
        col = pd.Series(np.arange(n), index=self.dates)
        ri = daily["meter_label"].map(row).to_numpy()
        ci = col.reindex(daily["date"]).to_numpy()

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
        self.zone = np.array([int(lab.split("-")[0].split(".")[1]) for lab in self.meters])

        cal = calendar_features(pd.Series(self.dates)).reset_index(drop=True)
        self.cal = {c: cal[c].to_numpy() for c in cal.columns}
        self.cal["days_from_ramadan_start"] = np.clip(self.cal["days_from_ramadan_start"], -60, 60)
        self.cal["days_from_eid_adha"] = np.clip(self.cal["days_from_eid_adha"], -30, 30)

        obs = weather.set_index("date")["t_mean"].reindex(self.dates).to_numpy()
        self.temp_obs = obs
        self.temp_normal = {}  # normal for each year, built from earlier years only
        for year in sorted(set(self.dates.year)):
            normal = climate_normal(weather, before_year=year)["normal_t_mean"]
            doy = np.clip(self.dates.dayofyear.to_numpy(), 1, 365)
            self.temp_normal[year] = normal.reindex(doy).to_numpy()

        self._cs, self._cn = self._cumsum(self.y)
        self._dow_cs, self._dow_cn = {}, {}
        dow = self.cal["dow"]
        for w in range(7):
            masked = np.where(dow[None, :] == w, self.y, np.nan)
            self._dow_cs[w], self._dow_cn[w] = self._cumsum(masked)
        self._ram = np.r_[0, np.cumsum(self.cal["ramadan"])]
        self._off = np.r_[0, np.cumsum(self.cal["eid"] | self.cal["national_holiday"])]
        self._tcs = np.r_[0, np.cumsum(np.nan_to_num(obs))]
        self._tcn = np.r_[0, np.cumsum(~np.isnan(obs))]
        self._last_meas = self._last_measurable_index()
        self.responses = ResponseBook(self)

    @staticmethod
    def _cumsum(values):
        ok = ~np.isnan(values)
        cs = np.concatenate([np.zeros((values.shape[0], 1)), np.cumsum(np.where(ok, values, 0), 1)], 1)
        cn = np.concatenate([np.zeros((values.shape[0], 1)), np.cumsum(ok, 1)], 1)
        return cs, cn

    def _last_measurable_index(self):
        idx = np.where(~np.isnan(self.y), np.arange(self.y.shape[1])[None, :], -1)
        return np.maximum.accumulate(idx, axis=1)

    def window_mean(self, a, b, cs=None, cn=None):
        """Mean of y over day indices [a, b) per meter; a/b may be arrays."""
        cs = self._cs if cs is None else cs
        cn = self._cn if cn is None else cn
        a = np.clip(a, 0, cs.shape[1] - 1)
        b = np.clip(b, 0, cs.shape[1] - 1)
        rows = np.arange(cs.shape[0])[:, None] if np.ndim(a) == 2 else np.arange(cs.shape[0])
        total = cs[rows, b] - cs[rows, a]
        count = cn[rows, b] - cn[rows, a]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(count > 0, total / np.maximum(count, 1), np.nan), count

    def index_of(self, day):
        return int((pd.Timestamp(day) - self.dates[0]).days)


def origin_rows(panel, t, horizons=MAX_HORIZON, meters=None):
    """Feature rows for forecast issue day index t, all meters x days ahead."""
    m = len(panel.meters)
    meters = np.arange(m) if meters is None else np.asarray(meters)
    k = np.arange(horizons)
    d = t + k                                            # target day indices
    n = len(panel.dates)

    lvl7, c7 = panel.window_mean(t - 7, t)
    lvl28, c28 = panel.window_mean(t - 28, t)
    last = panel._last_meas[:, t - 1] if t > 0 else np.full(m, -1)
    recent_start = np.maximum(last - 13, 0)
    lvl_last, c_last = panel.window_mean(recent_start, np.maximum(last + 1, 0))
    base = np.where(c7 >= 4, lvl7, np.where(c28 >= 10, lvl28, lvl_last))
    base_age = t - 1 - last  # days since the meter's last trusted day
    has_base = (last >= 0) & ~np.isnan(base)

    yesterday = panel.y[:, t - 1] if t > 0 else np.full(m, np.nan)
    two_days_ago = panel.y[:, t - 2] if t > 1 else np.full(m, np.nan)
    lvl3, c3 = panel.window_mean(t - 3, t)
    lvl3 = np.where(c3 >= 2, lvl3, np.nan)
    recent = np.log1p(panel.y[:, max(t - 28, 0):t])
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        volatility = np.nanstd(recent, axis=1) if recent.shape[1] else np.full(m, np.nan)

    dd = np.clip(d, 0, n - 1)
    dow_t = panel.cal["dow"][dd]
    dow_mean = np.full((m, horizons), np.nan)
    for w in range(7):
        cols = dow_t == w
        if cols.any():
            mean_w, cnt = panel.window_mean(t - 28, t, panel._dow_cs[w], panel._dow_cn[w])
            dow_mean[:, cols] = np.where(cnt >= 2, mean_w, np.nan)[:, None]

    # Same weekday one week before the target, only if that day is already known.
    lag7_idx = d - 7
    same_week = np.where(lag7_idx[None, :] < t, panel.y[:, np.clip(lag7_idx, 0, n - 1)], np.nan)

    # Last year's change from the base window to the target day (364 days keeps weekdays).
    ly_t, ly_tc = panel.window_mean((d - 364 - 3)[None, :].repeat(m, 0), (d - 364 + 4)[None, :].repeat(m, 0))
    ly_b, ly_bc = panel.window_mean(t - 364 - 7, t - 364)
    ly_ok = (ly_tc >= 3) & (ly_bc[:, None] >= 3) & (d[None, :] - 364 - 3 >= 0)
    ly_change = np.where(ly_ok, np.log1p(ly_t) - np.log1p(ly_b)[:, None], np.nan)

    year = panel.dates[min(t, n - 1)].year
    normal = panel.temp_normal[year]
    normal_target = normal[dd]
    normal_base = np.nanmean(normal[max(t - 7, 0):t]) if t > 0 else np.nan
    tc = panel._tcn[t] - panel._tcn[max(t - 7, 0)]
    recent_temp = (panel._tcs[t] - panel._tcs[max(t - 7, 0)]) / tc if tc > 0 else np.nan

    # Change implied by each meter's temperature curve: recent observed -> target normal.
    coefs = panel.responses.coefs_for(t)
    recent_obs = panel.temp_obs[max(t - 7, 0):t]
    recent_obs = recent_obs[~np.isnan(recent_obs)]
    recent_effect = hinge(recent_obs).mean(0) @ coefs.T if len(recent_obs) else np.full(m, np.nan)
    target_effect = effect(coefs, normal_target).T            # meters x horizons
    temp_change = target_effect - recent_effect[:, None]
    # Split into the seasonal part (normal -> normal) and the recent-weather anomaly.
    base_normal = normal[max(t - 7, 0):t]
    base_normal_effect = hinge(base_normal).mean(0) @ coefs.T if len(base_normal) else np.full(m, np.nan)
    seasonal_change = target_effect - base_normal_effect[:, None]
    anomaly_effect = base_normal_effect - recent_effect
    sensitivity = effect(coefs, np.array([38.0]))[0]

    def grid(values):
        return np.broadcast_to(values, (m, horizons))

    log_base = np.log1p(base)
    frame = {
        "meter_idx": grid(np.arange(m)[:, None]),
        "origin_idx": grid(np.full((1, 1), t)),
        "target_idx": grid(d[None, :]),
        "days_ahead": grid(k[None, :]),
        "meter_code": grid(np.arange(m)[:, None]),
        "base_kwh": grid(base[:, None]),
        "log_base": grid(log_base[:, None]),
        "base_age_days": grid(base_age[:, None]),
        "trend_7_vs_28": grid((np.log1p(lvl7) - np.log1p(lvl28))[:, None]),
        "yesterday_vs_base": grid((np.log1p(yesterday) - log_base)[:, None]),
        "two_days_ago_vs_base": grid((np.log1p(two_days_ago) - log_base)[:, None]),
        "last_3_days_vs_base": grid((np.log1p(lvl3) - log_base)[:, None]),
        "volatility_28": grid(volatility[:, None]),
        "dow_ratio": np.log1p(dow_mean) - np.log1p(lvl28)[:, None],
        "same_day_last_week_vs_base": np.log1p(same_week) - log_base[:, None],
        "last_year_change": ly_change,
        "last_year_available": ly_ok.astype(int),
        "base_ramadan_share": grid(np.full((1, 1), (panel._ram[t] - panel._ram[max(t - 7, 0)]) / 7.0)),
        "base_off_share": grid(np.full((1, 1), (panel._off[t] - panel._off[max(t - 7, 0)]) / 7.0)),
        "normal_temp_target": grid(normal_target[None, :]),
        "normal_temp_shift": grid((normal_target - normal_base)[None, :]),
        "recent_temp_anomaly": grid(np.full((1, 1), recent_temp - normal_base)),
        "recent_temp": grid(np.full((1, 1), recent_temp)),
        "temp_effect_change": temp_change,
        "seasonal_effect_change": seasonal_change,
        "weather_anomaly_effect": grid(anomaly_effect[:, None]),
        "temp_sensitivity": grid(sensitivity[:, None]),
        "has_base": grid(has_base[:, None]),
    }
    for c in ("dow", "weekend", "ramadan", "eid", "national_holiday", "off_day",
              "days_from_ramadan_start", "days_from_eid_adha"):
        frame[c] = grid(panel.cal[c][dd][None, :])
    out = pd.DataFrame({key: np.asarray(val)[meters].ravel() for key, val in frame.items()})
    valid_target = out["target_idx"].to_numpy() < n
    out = out[valid_target & out["has_base"].to_numpy()].drop(columns="has_base")
    tgt = out["target_idx"].to_numpy()
    mi = out["meter_idx"].to_numpy()
    out["actual_kwh"] = panel.y[mi, tgt]
    out["target"] = np.log1p(out["actual_kwh"]) - out["log_base"]
    return out.reset_index(drop=True)


def training_rows(panel, origin_indices, cutoff_idx, k_lo=0, k_hi=MAX_HORIZON):
    """Rows for days-ahead in [k_lo, k_hi) whose target was observed before ``cutoff_idx``."""
    frames = []
    for t in origin_indices:
        if t >= cutoff_idx or t < 8:
            continue
        rows = origin_rows(panel, t, horizons=min(k_hi, cutoff_idx - t))
        rows = rows[(rows["days_ahead"] >= k_lo) & (rows["target_idx"] < cutoff_idx)]
        frames.append(rows.dropna(subset=["target"]))
    return pd.concat(frames, ignore_index=True)
