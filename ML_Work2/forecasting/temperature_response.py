"""How strongly each meter's consumption responds to Riyadh temperature.

For every meter we fit   log(kWh + 1) = level + sum_k b_k * max(T - k, 0)
                                       + weekend / Ramadan / Eid / national-day terms
with temperature hinges at 15, 20, 25, 30 and 35 C (cooling load grows faster
in extreme heat). Each meter's curve is pulled toward the curve of all meters
together (ridge shrinkage), so a meter with little history still gets a
sensible response.

The fit is redone at every month start using only data before that month, and
a forecast issued in a month uses that month's curves - never later data.
The model then receives "how much should consumption change because the
season moves from the recent temperature to the target day's normal".
"""
import numpy as np

HINGES = np.array([15.0, 20.0, 25.0, 30.0, 35.0])
SHRINK_DAYS = 90.0   # a meter's own data must outweigh ~90 days of the pooled curve


def hinge(temp):
    temp = np.asarray(temp, dtype=float)
    return np.maximum(temp[..., None] - HINGES, 0.0)


def effect(coefs, temp):
    """Temperature effect in log-kWh: shape (len(temp), meters)."""
    return hinge(temp) @ coefs.T


def _design(panel, upto):
    temp = panel.temp_obs[:upto]
    x_temp = hinge(temp)
    controls = np.column_stack([panel.cal[c][:upto] for c in ("weekend", "ramadan", "eid", "national_holiday")])
    return np.column_stack([x_temp, controls]).astype(float), ~np.isnan(temp)


def fit(panel, cutoff_idx):
    """Per-meter hinge slopes (meters x 5) from data before ``cutoff_idx``."""
    m = len(panel.meters)
    coefs = np.zeros((m, len(HINGES)))
    if cutoff_idx < 60:
        return coefs
    x_all, temp_ok = _design(panel, cutoff_idx)
    y_all = np.log1p(panel.y[:, :cutoff_idx])
    stacked_x, stacked_y, per_meter = [], [], []
    for i in range(m):
        ok = temp_ok & ~np.isnan(y_all[i])
        if ok.sum() < 30:
            per_meter.append(None)
            continue
        x = x_all[ok] - x_all[ok].mean(0)
        y = y_all[i, ok] - y_all[i, ok].mean()
        stacked_x.append(x)
        stacked_y.append(y)
        per_meter.append((x, y))
    if not stacked_x:
        return coefs
    xp, yp = np.vstack(stacked_x), np.concatenate(stacked_y)
    scale = (xp ** 2).mean(0) + 1e-9
    pooled = np.linalg.solve(xp.T @ xp + 1e-3 * np.diag(scale) * len(yp), xp.T @ yp)
    penalty = SHRINK_DAYS * np.diag(scale)
    for i, item in enumerate(per_meter):
        if item is None:
            coefs[i] = pooled[: len(HINGES)]
            continue
        x, y = item
        b = np.linalg.solve(x.T @ x + penalty, x.T @ y + penalty @ pooled)
        coefs[i] = b[: len(HINGES)]
    return coefs


class ResponseBook:
    """Curves fitted at each month start; lookup by forecast issue day."""

    def __init__(self, panel):
        months = panel.dates.to_period("M")
        self.month_start_idx = {}
        for idx, (period, first) in enumerate(zip(months, np.r_[True, months[1:] != months[:-1]])):
            if first:
                self.month_start_idx[period] = idx
        self.curves = {p: fit(panel, i) for p, i in self.month_start_idx.items()}
        self.months = months

    def coefs_for(self, origin_idx):
        return self.curves[self.months[min(origin_idx, len(self.months) - 1)]]
