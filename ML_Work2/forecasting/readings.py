"""Raw meter readings and the gold-layer consumption rule.

Sources:
  * gold.energy_raw            - every reading for all 31 meters up to 2026-07-11
  * input_excel/*.xlsx         - telemetry exports from 2026-07-12 onwards

`apply_gold_rule` reproduces silver.fact_energy_consumption_validated:
startup exclusions, isolated low/high glitches dropped, consumption equal to the
cumulative register minus the previous valid reading, and LONG_GAP for any
interval of 15 minutes or more. `check_rule_against_gold` proves the
reproduction day by day before it is trusted for Zone 13 and the Excel period.
"""
import json
import re

import numpy as np
import pandas as pd

from . import config
from .database import connect

DB_CACHE = config.DATA_DIR / "raw_readings_database.parquet"
EXCEL_CACHE = config.DATA_DIR / "raw_readings_excel.parquet"
IDENTITY_AUDIT = config.DATA_DIR / "excel_meter_identity.json"


def load_meters(conn):
    """Meter label, zone and key for all 31 meters (resolved once, never joined to facts)."""
    sql = """
        SELECT m.energy_meter_key::text AS meter_key, m.meter_label, z.zone_name
        FROM gold.meter m LEFT JOIN gold.zone z ON z.zone_key = m.zone_key
        ORDER BY m.meter_label
    """
    return _query(conn, sql)


def _query(conn, sql, params=None):
    with conn.cursor() as cur:
        cur.execute(sql, params)
        cols = [d.name for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)


def load_database_readings(refresh=False):
    """Every raw reading (timestamp + cumulative kWh) for all meters, cached."""
    if DB_CACHE.exists() and not refresh:
        return pd.read_parquet(DB_CACHE)
    with connect() as conn:
        meters = load_meters(conn)
        frames = []
        for row in meters.itertuples():
            readings = _query(
                conn,
                """SELECT reading_timestamp_utc AS ts_utc, active_energy_cumulative_kwh AS kwh
                   FROM gold.energy_raw WHERE energy_meter_key = %s
                   ORDER BY reading_timestamp_utc""",
                (row.meter_key,),
            )
            readings["meter_label"] = row.meter_label
            frames.append(readings)
    out = pd.concat(frames, ignore_index=True)
    out["ts_utc"] = pd.to_datetime(out["ts_utc"])
    out["kwh"] = out["kwh"].astype(float)
    out["source"] = "database"
    out.to_parquet(DB_CACHE, index=False)
    meters.to_csv(config.DATA_DIR / "meters.csv", index=False)
    return out


def load_meter_table():
    path = config.DATA_DIR / "meters.csv"
    if not path.exists():
        with connect() as conn:
            load_meters(conn).to_csv(path, index=False)
    return pd.read_csv(path)


_FILE_LABEL = re.compile(r"_(ZN\.\d+-PB(?:\.\d+)?)_child_(\d+)_ch3pm_")


def _read_excel_file(path):
    frame = pd.read_excel(path, usecols=["timestamp", "kWh"], dtype=str, engine="openpyxl")
    frame = frame.dropna()
    ts = pd.to_datetime(pd.to_numeric(frame["timestamp"], errors="coerce"), unit="ms")
    kwh = pd.to_numeric(frame["kWh"], errors="coerce")
    out = pd.DataFrame({"ts_utc": ts, "kwh": kwh}).dropna()
    return out.sort_values("ts_utc").reset_index(drop=True)


def load_excel_readings(db_readings, refresh=False):
    """Parse every Excel export and resolve which meter it belongs to.

    The file name is checked against register continuity: the first Excel
    reading must continue the meter's last database reading. A file whose name
    does not continue its own meter is matched to the unclaimed meter it does
    continue (this is how the ZN.13 "child_2" export is identified).
    """
    if EXCEL_CACHE.exists() and IDENTITY_AUDIT.exists() and not refresh:
        return pd.read_parquet(EXCEL_CACHE)

    last = (db_readings.sort_values("ts_utc").groupby("meter_label")
            .agg(last_ts=("ts_utc", "last"), last_kwh=("kwh", "last")))
    files = sorted(config.INPUT_EXCEL.glob("*.xlsx"))
    if not files:
        print("No Excel files in input_excel/: using database readings only.")
        return pd.DataFrame({"meter_label": pd.Series(dtype=str),
                             "ts_utc": pd.Series(dtype="datetime64[ns]"),
                             "kwh": pd.Series(dtype=float),
                             "source": pd.Series(dtype=str)})
    parsed, audit = {}, []
    for path in files:
        match = _FILE_LABEL.search(path.name)
        if not match:
            raise ValueError(f"Cannot read a meter name from {path.name}")
        parsed[path.name] = (match.group(1), int(match.group(2)), _read_excel_file(path))

    def continuity(label, frame):
        first = frame.iloc[0]
        hours = (first.ts_utc - last.loc[label, "last_ts"]).total_seconds() / 3600
        step = first.kwh - last.loc[label, "last_kwh"]
        return step, hours

    claimed, assignment = set(), {}
    # Pass 1: names that continue their own meter's register.
    for name, (short, child, frame) in parsed.items():
        label = f"{short}_child__{child}_ch3pm"
        if label in last.index:
            step, hours = continuity(label, frame)
            if 0 <= step <= 60 * max(hours, 1):
                assignment[name] = (label, step, hours, "file name, register continues")
                claimed.add(label)
    # Pass 2: anything left is matched by continuity among unclaimed meters.
    for name, (short, child, frame) in parsed.items():
        if name in assignment:
            continue
        best = None
        for label in last.index.difference(sorted(claimed)):
            step, hours = continuity(label, frame)
            if step >= 0 and (best is None or step < best[1]):
                best = (label, step, hours)
        if best is None or best[1] > 60 * max(best[2], 1):
            raise ValueError(f"No meter continues the register of {name}")
        assignment[name] = (best[0], best[1], best[2],
                            f"register continuity (file named {short} child_{child})")
        claimed.add(best[0])

    frames = []
    for name, (label, step, hours, how) in assignment.items():
        frame = parsed[name][2].copy()
        frame["meter_label"] = label
        frames.append(frame)
        audit.append({"file": name, "meter_label": label, "matched_by": how,
                      "first_excel_kwh": float(parsed[name][2].iloc[0].kwh),
                      "last_database_kwh": float(last.loc[label, "last_kwh"]),
                      "kwh_across_handover": round(float(step), 2),
                      "hours_across_handover": round(float(hours), 2),
                      "excel_rows": int(len(parsed[name][2]))})
    missing = sorted(set(last.index) - claimed)
    out = pd.concat(frames, ignore_index=True)
    out["source"] = "excel"
    out.to_parquet(EXCEL_CACHE, index=False)
    IDENTITY_AUDIT.write_text(json.dumps({"files": audit, "meters_without_excel": missing}, indent=2))
    return out


def apply_gold_rule(readings):
    """Per-reading validated consumption, reproducing the gold layer.

    Returns one row per kept reading with: meter_label, ts_utc, kwh,
    consumption_kwh, gap_minutes, interval_quality, status.
    """
    results = []
    for label, frame in readings.sort_values(["meter_label", "ts_utc"]).groupby("meter_label", sort=False):
        frame = frame.reset_index(drop=True)
        ts = frame["ts_utc"].to_numpy()
        c = frame["kwh"].to_numpy(dtype=float)
        status = np.full(len(c), "KEEP", dtype=object)

        cutoff = config.STARTUP_EXCLUDED_BEFORE_UTC.get(label)
        if cutoff is not None:
            status[ts < np.datetime64(cutoff)] = "STARTUP_EXCLUDED"

        prev = np.r_[np.nan, c[:-1]]
        nxt = np.r_[c[1:], np.nan]
        live = status == "KEEP"
        low_bad = live & (c < prev) & (nxt >= prev)
        high_bad = live & (c > nxt) & (nxt >= prev)
        status[low_bad] = "ISOLATED_LOW_BAD"
        status[high_bad] = "ISOLATED_HIGH_BAD"

        keep = status == "KEEP"
        kept = frame.loc[keep, ["meter_label", "ts_utc", "kwh"]].copy()
        kept["consumption_kwh"] = kept["kwh"].diff()
        kept["gap_minutes"] = kept["ts_utc"].diff().dt.total_seconds() / 60
        kept["status"] = "KEEP"
        first = kept.index[0]
        kept.loc[first, "status"] = "FIRST_VALID_READING"
        # A decrement no rule explains is never booked as negative energy.
        unresolved = kept["consumption_kwh"] < 0
        kept.loc[unresolved, "status"] = "UNRESOLVED_DECREMENT"
        kept.loc[unresolved, "consumption_kwh"] = 0.0
        implied_kw = kept["consumption_kwh"] / (kept["gap_minutes"] / 60.0)
        jump = (kept["gap_minutes"] < config.LONG_GAP_MINUTES) & (implied_kw > config.MAX_PLAUSIBLE_METER_KW)
        kept.loc[jump, "status"] = "REGISTER_JUMP"
        kept.loc[jump, "consumption_kwh"] = 0.0
        kept["interval_quality"] = np.where(kept["gap_minutes"] >= config.LONG_GAP_MINUTES,
                                            "LONG_GAP", "NORMAL_INTERVAL")
        kept.loc[first, "interval_quality"] = "FIRST_VALID_READING"
        kept["dropped_readings"] = int((~keep).sum())
        results.append(kept)
    out = pd.concat(results, ignore_index=True)
    out["date"] = (out["ts_utc"] + pd.Timedelta(hours=config.SAUDI_OFFSET_HOURS)).dt.normalize()
    return out


def gold_daily_from_database():
    """Daily kWh exactly as gold.energy_consumption_validated books it (29 meters)."""
    with connect() as conn:
        out = _query(conn, """
            SELECT meter_label, reading_date_saudi AS date,
                   sum(validated_kwh_consumption) AS gold_kwh
            FROM gold.energy_consumption_validated
            GROUP BY 1, 2""")
    out["date"] = pd.to_datetime(out["date"])
    out["gold_kwh"] = out["gold_kwh"].astype(float)
    return out


def check_rule_against_gold(validated):
    """Compare our reproduction with gold for every meter-day gold contains."""
    gold = gold_daily_from_database()
    ours = (validated.groupby(["meter_label", "date"])["consumption_kwh"]
            .sum(min_count=1).rename("our_kwh").reset_index())
    merged = gold.merge(ours, on=["meter_label", "date"], how="left")
    merged["our_kwh"] = merged["our_kwh"].fillna(0.0)
    merged["gold_kwh"] = merged["gold_kwh"].fillna(0.0)
    merged["diff"] = (merged["our_kwh"] - merged["gold_kwh"]).abs()
    return {
        "meter_days_compared": int(len(merged)),
        "meter_days_exact_within_0.01_kwh": int((merged["diff"] <= 0.01).sum()),
        "max_abs_difference_kwh": float(merged["diff"].max()),
        "gold_total_kwh": round(float(merged["gold_kwh"].sum()), 1),
        "our_total_kwh_same_meter_days": round(float(merged["our_kwh"].sum()), 1),
        "worst_rows": merged.sort_values("diff", ascending=False).head(10)
                      .assign(date=lambda d: d["date"].dt.strftime("%Y-%m-%d"))
                      .to_dict("records"),
    }
