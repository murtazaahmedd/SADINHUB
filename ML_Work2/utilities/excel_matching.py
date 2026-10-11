"""Tie each Excel export to the meter whose register it continues.

The file name is checked against register continuity: the first Excel reading
must continue the meter's last database reading. A file whose name does not
continue its own meter is matched to the unclaimed meter it does continue
(this is how a misleadingly named export is identified).
"""
from typing import Dict, List, Set, Tuple

import pandas as pd

from constants.data_rules import EXCEL_FILE_LABEL, EXCEL_MAX_CONTINUITY_KWH_PER_HOUR
from errors import Errors
from utilities.guard import require, traced

Parsed = Dict[str, Tuple[str, int, pd.DataFrame]]
Assignment = Dict[str, Tuple[str, float, float, str]]


@traced
def parse_file_name(name: str) -> Tuple[str, int]:
    """
    #Feature: Excel matching
    #Scenario: Meter short name and child number from an export's file name
    #Complexity: Time O(len(name)) - linear

    Args:
        name: Export file name.

    Returns:
        (short name such as "ZN.13-PB.1", child number).

    Raises:
        ValueError: When name is None.
        Errors.Misc.MeterMappingError: When no meter name can be read from it.
    """
    require("parse_file_name", "name", name)
    match = EXCEL_FILE_LABEL.search(name)
    if not match:
        raise Errors.Misc.MeterMappingError(
            f"[parse_file_name] Cannot read a meter name from {name}. Invariant violated: export names contain "
            f"'_ZN.<zone>-PB[.<n>]_child_<n>_ch3pm_'. Received: {name}. Fix: rename the file or update "
            f"constants.data_rules.EXCEL_FILE_LABEL.")
    return match.group(1), int(match.group(2))


@traced
def last_readings(db_readings: pd.DataFrame) -> pd.DataFrame:
    """
    #Feature: Excel matching
    #Scenario: Each meter's last database timestamp and register value
    #Complexity: Time O(n log n) - sort

    Args:
        db_readings: Database readings (meter_label, ts_utc, kwh).

    Returns:
        DataFrame indexed by meter_label with last_ts and last_kwh.

    Raises:
        ValueError: When db_readings is None.
    """
    require("last_readings", "db_readings", db_readings)
    return (db_readings.sort_values("ts_utc").groupby("meter_label")
            .agg(last_ts=("ts_utc", "last"), last_kwh=("kwh", "last")))


@traced
def continuity(label: str, frame: pd.DataFrame, last: pd.DataFrame) -> Tuple[float, float]:
    """
    #Feature: Excel matching
    #Scenario: kWh step and hours between a meter's last database reading and an export's first
    #Complexity: Time O(1) - constant

    Args:
        label: Meter label.
        frame: Export readings (ts_utc, kwh).
        last: Output of last_readings.

    Returns:
        (kWh step, hours elapsed).

    Raises:
        ValueError: When an argument is None.
    """
    require("continuity", "label", label)
    require("continuity", "frame", frame)
    first = frame.iloc[0]
    hours = (first.ts_utc - last.loc[label, "last_ts"]).total_seconds() / 3600
    step = first.kwh - last.loc[label, "last_kwh"]
    return step, hours


@traced
def match_exports(parsed: Parsed, last: pd.DataFrame) -> Tuple[Assignment, Set[str]]:
    """
    #Feature: Excel matching
    #Scenario: Decide which meter each export belongs to
    #Complexity: Time O(f*m) - files times meters

    Pass 1 trusts a name that continues its own meter's register. Pass 2 matches
    every remaining file to the unclaimed meter whose register it continues.

    Args:
        parsed: file name -> (short name, child number, readings).
        last: Output of last_readings.

    Returns:
        (file name -> (meter label, kWh step, hours, how matched), claimed meter labels).

    Raises:
        ValueError: When an argument is None.
        Errors.Misc.MeterMappingError: When no meter continues a file's register.
    """
    require("match_exports", "parsed", parsed)
    require("match_exports", "last", last)
    claimed: Set[str] = set()
    assignment: Assignment = {}
    for name, (short, child, frame) in parsed.items():
        label = f"{short}_child__{child}_ch3pm"
        if label in last.index:
            step, hours = continuity(label, frame, last)
            if 0 <= step <= EXCEL_MAX_CONTINUITY_KWH_PER_HOUR * max(hours, 1):
                assignment[name] = (label, step, hours, "file name, register continues")
                claimed.add(label)
    for name, (short, child, frame) in parsed.items():
        if name in assignment:
            continue
        best = _best_unclaimed(frame, last, claimed)
        if best is None or best[1] > EXCEL_MAX_CONTINUITY_KWH_PER_HOUR * max(best[2], 1):
            raise Errors.Misc.MeterMappingError(
                f"[match_exports] No meter continues the register of {name}. Invariant violated: every export "
                f"continues exactly one meter's last database reading. Received: no match. "
                f"Fix: check the file is for this facility and covers the period right after the database.")
        assignment[name] = (best[0], best[1], best[2], f"register continuity (file named {short} child_{child})")
        claimed.add(best[0])
    return assignment, claimed


@traced
def _best_unclaimed(frame: pd.DataFrame, last: pd.DataFrame, claimed: Set[str]):
    """
    #Feature: Excel matching
    #Scenario: The unclaimed meter whose register the export continues most closely
    #Complexity: Time O(m) - meters

    Args:
        frame: Export readings.
        last: Output of last_readings.
        claimed: Meter labels already matched.

    Returns:
        (label, step, hours) of the best candidate, or None.

    Raises:
        ValueError: When an argument is None.
    """
    require("_best_unclaimed", "frame", frame)
    require("_best_unclaimed", "claimed", claimed)
    best = None
    for label in last.index.difference(sorted(claimed)):
        step, hours = continuity(label, frame, last)
        if step >= 0 and (best is None or step < best[1]):
            best = (label, step, hours)
    return best


@traced
def build_audit(assignment: Assignment, parsed: Parsed, last: pd.DataFrame) -> List[dict]:
    """
    #Feature: Excel matching
    #Scenario: One audit row per export showing how it was matched
    #Complexity: Time O(f) - files

    Args:
        assignment: Output of match_exports.
        parsed: file name -> (short, child, readings).
        last: Output of last_readings.

    Returns:
        Audit rows (file, meter, how, handover step and hours, row count).

    Raises:
        ValueError: When an argument is None.
    """
    require("build_audit", "assignment", assignment)
    return [{"file": name, "meter_label": label, "matched_by": how,
             "first_excel_kwh": float(parsed[name][2].iloc[0].kwh),
             "last_database_kwh": float(last.loc[label, "last_kwh"]),
             "kwh_across_handover": round(float(step), 2), "hours_across_handover": round(float(hours), 2),
             "excel_rows": int(len(parsed[name][2]))}
            for name, (label, step, hours, how) in assignment.items()]
