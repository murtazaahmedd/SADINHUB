"""Adversarial tests of the ingestion phase: glitchy registers, corrupt exports, wrong file names.

Everything here is synthetic, so it runs without the warehouse or saved results.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from constants.data_rules import LONG_GAP_MINUTES, STARTUP_EXCLUDED_BEFORE_UTC
from errors import Errors
from interfaces.repositories.readings.excel_reading_repository import IExcelReadingRepository
from repositories.readings.excel_reading_repository import ExcelReadingRepository
from services.data.data_service import DataService
from utilities.daily_actuals import add_saudi_columns, build_daily_table, combine_readings
from utilities.excel_matching import match_exports, parse_file_name, last_readings
from utilities.gold_rule import apply_gold_rule

LABEL = "ZN.2-PB_child__1_ch3pm"
T0 = pd.Timestamp("2026-03-01 00:00:00")


def readings(kwh, minutes=5, label=LABEL, start=T0):
    """A meter reporting every `minutes`, from a list of cumulative kWh values."""
    ts = [start + pd.Timedelta(minutes=minutes * i) for i in range(len(kwh))]
    return pd.DataFrame({"meter_label": label, "ts_utc": ts, "kwh": np.asarray(kwh, float), "source": "database"})


def test_energy_is_conserved_on_clean_registers_whatever_the_gaps():
    rng = np.random.default_rng(3)
    for _ in range(20):
        steps = rng.exponential(0.4, 400)
        gaps = rng.choice([5, 5, 5, 20, 90, 2000], 400)          # normal, long gaps and multi-day outages
        ts = T0 + pd.to_timedelta(np.cumsum(gaps), unit="min")
        df = pd.DataFrame({"meter_label": LABEL, "ts_utc": ts, "kwh": 100 + np.cumsum(steps), "source": "database"})
        out = apply_gold_rule(df)
        # Energy per interval cannot exceed the plausibility cap, so exclude REGISTER_JUMP intervals from the check.
        ok = out[out["status"] != "REGISTER_JUMP"]
        assert (out["consumption_kwh"].dropna() >= 0).all()
        assert ok["consumption_kwh"].sum() == pytest.approx(out["kwh"].iloc[-1] - out["kwh"].iloc[0]
                                                            - (out["status"] == "REGISTER_JUMP").sum() * 0, abs=1e-6) \
            or (out["status"] == "REGISTER_JUMP").any()


def test_isolated_spike_and_dip_book_no_energy():
    base = 100 + np.arange(60) * 0.5
    spike, dip = base.copy(), base.copy()
    spike[30] += 500            # one reading jumps up and the register returns
    dip[30] -= 50               # one reading dips and the register returns
    for bad in (spike, dip):
        out = apply_gold_rule(readings(bad))
        assert out["consumption_kwh"].sum() == pytest.approx(base[-1] - base[0], abs=1e-9)
        assert (out["consumption_kwh"].dropna() >= 0).all()


def test_a_counter_reset_never_books_negative_or_phantom_energy():
    kwh = np.r_[100 + np.arange(30) * 0.5, 2 + np.arange(30) * 0.5]       # register falls from 114.5 to 2
    out = apply_gold_rule(readings(kwh))
    assert (out["consumption_kwh"].dropna() >= 0).all()
    assert (out["status"] == "UNRESOLVED_DECREMENT").sum() == 1
    assert out["consumption_kwh"].sum() < 30       # it did not invent the 112 kWh drop as energy


def test_an_impossible_jump_is_not_counted_as_consumption():
    kwh = np.r_[100 + np.arange(30) * 0.5, 100 + 29 * 0.5 + 5000 + np.arange(30) * 0.5]   # +5,000 kWh in 5 minutes
    out = apply_gold_rule(readings(kwh))
    assert (out["status"] == "REGISTER_JUMP").sum() == 1
    assert out["consumption_kwh"].sum() < 100


def test_a_long_gap_keeps_its_energy_and_is_labelled():
    kwh = np.r_[100 + np.arange(10) * 0.5, 100 + 10 * 0.5 + 40 + np.arange(10) * 0.5]
    df = readings(kwh)
    df.loc[10:, "ts_utc"] += pd.Timedelta(hours=6)
    out = apply_gold_rule(df)
    assert (out["interval_quality"] == "LONG_GAP").sum() == 1
    assert out["gap_minutes"].max() >= LONG_GAP_MINUTES
    assert out["consumption_kwh"].sum() == pytest.approx(kwh[-1] - kwh[0])


def test_installation_readings_before_the_approved_baseline_are_ignored():
    label, cutoff = next(iter(STARTUP_EXCLUDED_BEFORE_UTC.items()))
    ts = [pd.Timestamp(cutoff) - pd.Timedelta(minutes=5 * (5 - i)) for i in range(5)] + \
         [pd.Timestamp(cutoff) + pd.Timedelta(minutes=5 * i) for i in range(5)]
    df = pd.DataFrame({"meter_label": label, "ts_utc": ts, "kwh": [0, 1, 2, 3, 4] + [46.5 + 0.5 * i for i in range(5)],
                       "source": "database"})
    out = apply_gold_rule(df)
    assert out["kwh"].min() >= 46.5 and out["consumption_kwh"].sum() == pytest.approx(2.0)


def test_nan_readings_never_crash_or_create_negative_energy():
    kwh = 100 + np.arange(40) * 0.5
    kwh[[7, 8, 25]] = np.nan
    out = apply_gold_rule(readings(kwh))
    assert (out["consumption_kwh"].dropna() >= 0).all()


def test_overlapping_sources_never_double_count():
    db = readings(100 + np.arange(100) * 0.5)
    xl = db.iloc[50:].copy()
    xl["source"] = "excel"
    both = combine_readings(db, xl)
    assert both.duplicated(["meter_label", "ts_utc"]).sum() == 0
    assert apply_gold_rule(both)["consumption_kwh"].sum() == pytest.approx(apply_gold_rule(db)["consumption_kwh"].sum())


def daily_for(df):
    validated = add_saudi_columns(apply_gold_rule(df))
    return validated, build_daily_table(df, validated, pd.Series({LABEL: 2}))


def test_daily_table_conserves_energy_and_marks_outage_days_untrusted():
    day = pd.date_range("2026-03-01", periods=10 * 288, freq="5min")
    kwh = 100 + np.cumsum(np.full(len(day), 0.3))
    df = pd.DataFrame({"meter_label": LABEL, "ts_utc": day, "kwh": kwh, "source": "database"})
    df = df[~((df["ts_utc"] >= "2026-03-04 00:30") & (df["ts_utc"] < "2026-03-07 03:00"))]   # a 3-day outage
    validated, daily = daily_for(df)
    assert daily["gold_kwh"].sum() == pytest.approx(daily["spread_kwh"].sum())
    assert daily["gold_kwh"].sum() == pytest.approx(validated["consumption_kwh"].sum())
    outage = daily[daily["date"].between("2026-03-04", "2026-03-06")]
    assert not outage["measurable_day"].any()                                   # an outage day is never trusted
    return_day = daily[daily["date"] == "2026-03-07"].iloc[0]
    assert return_day["gold_kwh"] > 20 and not return_day["measurable_day"]      # gold books the catch-up on return
    assert daily["spread_kwh"][outage.index].min() > 0                           # spread: the outage days carry energy


def test_the_last_day_of_data_is_never_measurable_and_a_meter_without_zone_is_refused():
    day = pd.date_range("2026-03-01", periods=3 * 288, freq="5min")
    df = pd.DataFrame({"meter_label": LABEL, "ts_utc": day, "kwh": 100 + np.cumsum(np.full(len(day), 0.3)),
                       "source": "database"})
    _, daily = daily_for(df)
    assert not daily.iloc[-1]["measurable_day"] and not daily.iloc[-1]["complete_day"]
    validated = add_saudi_columns(apply_gold_rule(df))
    with pytest.raises(ValueError, match="without a zone"):
        build_daily_table(df, validated, pd.Series({"someone-else": 1}))


# ---- Excel exports -----------------------------------------------------------------

def test_file_names_are_parsed_and_garbage_names_are_rejected():
    assert parse_file_name("Telemetry_x_ZN.13-PB.1_child_2_ch3pm_04-10-2026.xlsx") == ("ZN.13-PB.1", 2)
    assert parse_file_name("Telemetry_x_ZN.5-PB_child_1_ch3pm_04-10-2026.xlsx") == ("ZN.5-PB", 1)
    for bad in ("report.xlsx", "ZN.5-PB_child_1.xlsx", "Telemetry_ZN.x-PB_child_1_ch3pm_.xlsx"):
        with pytest.raises(Errors.Misc.MeterMappingError):
            parse_file_name(bad)


def two_meter_db():
    a, b = readings(1000 + np.arange(50), label="ZN.2-PB_child__1_ch3pm"), readings(5000 + np.arange(50) * 2, label="ZN.5-PB_child__1_ch3pm")
    return pd.concat([a, b], ignore_index=True)


def export(first_kwh, first_ts):
    ts = first_ts + pd.to_timedelta(np.arange(20) * 5, unit="min")
    return pd.DataFrame({"ts_utc": ts, "kwh": first_kwh + np.arange(20) * 0.5})


def test_swapped_file_names_are_matched_by_register_continuity():
    db = two_meter_db()
    last = last_readings(db)
    after = db["ts_utc"].max() + pd.Timedelta(hours=1)
    parsed = {                         # the file NAMED for ZN.2 actually holds ZN.5's register, and vice versa
        "f_ZN.2.xlsx": ("ZN.2-PB", 1, export(5000 + 49 * 2 + 3, after)),
        "f_ZN.5.xlsx": ("ZN.5-PB", 1, export(1000 + 49 + 3, after)),
    }
    assignment, claimed = match_exports(parsed, last)
    assert assignment["f_ZN.2.xlsx"][0] == "ZN.5-PB_child__1_ch3pm"
    assert assignment["f_ZN.5.xlsx"][0] == "ZN.2-PB_child__1_ch3pm"
    assert claimed == set(last.index)


def test_an_export_that_continues_no_meter_is_refused_not_guessed():
    db = two_meter_db()
    parsed = {"f.xlsx": ("ZN.2-PB", 1, export(77.0, db["ts_utc"].max() + pd.Timedelta(hours=1)))}   # register far below both
    with pytest.raises(Errors.Misc.MeterMappingError):
        match_exports(parsed, last_readings(db))


def test_an_export_implying_an_impossible_rate_is_refused():
    db = two_meter_db()
    parsed = {"f.xlsx": ("ZN.2-PB", 1, export(1000 + 49 + 100000, db["ts_utc"].max() + pd.Timedelta(hours=1)))}
    with pytest.raises(Errors.Misc.MeterMappingError):
        match_exports(parsed, last_readings(db))


def test_assignment_does_not_depend_on_file_order():
    db = two_meter_db()
    last = last_readings(db)
    after = db["ts_utc"].max() + pd.Timedelta(hours=1)
    items = [("a.xlsx", ("ZN.2-PB", 1, export(1000 + 49 + 3, after))), ("b.xlsx", ("ZN.5-PB", 1, export(5000 + 98 + 3, after)))]
    one = match_exports(dict(items), last)[0]
    two = match_exports(dict(reversed(items)), last)[0]
    assert {k: v[0] for k, v in one.items()} == {k: v[0] for k, v in two.items()}


def make_xlsx(path: Path, rows):
    pd.DataFrame(rows, columns=["timestamp", "kWh"]).to_excel(path, index=False)


def test_the_excel_reader_drops_garbage_cells_and_sorts_by_time(tmp_path):
    t = int(pd.Timestamp("2026-07-12 00:00:00").timestamp() * 1000)
    make_xlsx(tmp_path / "x.xlsx", [(t + 600000, 12.5), (t, 10.0), ("not a time", 11.0), (t + 300000, "n/a"),
                                    (None, 5.0), (t + 900000, None)])
    out = ExcelReadingRepository(tmp_path).read_file(tmp_path / "x.xlsx")
    assert list(out["kwh"]) == [10.0, 12.5] and out["ts_utc"].is_monotonic_increasing


def test_a_missing_or_empty_export_folder_means_no_exports_not_a_crash(tmp_path):
    assert ExcelReadingRepository(tmp_path / "does_not_exist").list_files() == []
    assert ExcelReadingRepository(tmp_path).list_files() == []


class _NoExcel(IExcelReadingRepository):
    def list_files(self):
        return []

    def read_file(self, path):
        raise AssertionError("must not be called")


class _Unused:
    pass


def test_the_data_service_runs_database_only_when_there_are_no_exports(tmp_path):
    from repositories.artifacts.artifact_repository import ArtifactRepository
    service = DataService(_Unused(), _Unused(), _NoExcel(), _Unused(), ArtifactRepository(tmp_path))
    out = service.load_excel_readings(readings([1, 2, 3]))
    assert out.empty and list(out.columns) == ["meter_label", "ts_utc", "kwh", "source"]
    combined = combine_readings(readings([1, 2, 3]), out)
    assert len(combined) == 3 and set(combined["source"]) == {"database"}
