"""Bill rules: kWh -> SAR for one meter and one 30-day billing cycle.

Rules (given by the facility, commercial tariff; numbers live in constants/tariff.py):
  * Billing cycles are 30 days counted from 1 January of the forecast year.
  * Energy charge per meter per cycle: the first 6,000 kWh at the lower rate,
    every kWh above that at the higher rate.
  * Plus a fixed monthly service fee set by the meter's circuit-breaker size.
  * Bill = (energy charge + service fee) x (1 + VAT).
"""
from datetime import date, timedelta
from typing import Dict, List, Tuple

import pandas as pd

from constants.tariff import (CYCLE_DAYS, RATE_ABOVE_LIMIT, RATE_UP_TO_LIMIT, SERVICE_FEE_CLASSES, TIER_LIMIT_KWH,
                              VAT_RATE)
from utilities.guard import require, traced


@traced
def service_class(amps: float) -> Tuple[str, float]:
    """
    #Feature: Bill forecast
    #Scenario: Service-fee class and monthly fee for a breaker size
    #Complexity: Time O(c) - classes

    Args:
        amps: Circuit-breaker rating in amps.

    Returns:
        (class name, monthly fee in SAR).

    Raises:
        ValueError: When amps is None or outside every class.
    """
    require("service_class", "amps", amps)
    for low, high, name, fee in SERVICE_FEE_CLASSES:
        if low <= amps < high:
            return name, fee
    raise ValueError(f"[service_class] Breaker rating {amps} A is outside every service-fee class. "
                     f"Invariant violated: every rating maps to a class. Received: {amps}. "
                     f"Fix: add a class to constants/tariff.py.")


@traced
def bill(kwh: float, fee: float) -> Dict[str, float]:
    """
    #Feature: Bill forecast
    #Scenario: All parts of one meter's bill for one cycle
    #Complexity: Time O(1) - constant

    Every money amount is rounded to the halala, and the bill is the exact sum of
    its rounded parts, so all totals on the dashboard add up.

    Args:
        kwh: Energy in the cycle (negative values count as zero).
        fee: Monthly service fee in SAR.

    Returns:
        Dict with kwh, the two tiers, the energy charges, fee, VAT and bill_sar.

    Raises:
        ValueError: When an argument is None.
    """
    require("bill", "kwh", kwh)
    require("bill", "fee", fee)
    kwh = max(float(kwh), 0.0)
    low = min(kwh, TIER_LIMIT_KWH)
    high = max(kwh - TIER_LIMIT_KWH, 0.0)
    energy_low = round(low * RATE_UP_TO_LIMIT, 2)
    energy_high = round(high * RATE_ABOVE_LIMIT, 2)
    fee = round(float(fee), 2)
    vat = round((energy_low + energy_high + fee) * VAT_RATE, 2)
    return {"kwh": kwh, "kwh_at_022": low, "kwh_at_032": high, "energy_022_sar": energy_low,
            "energy_032_sar": energy_high, "service_fee_sar": fee, "vat_sar": vat,
            "bill_sar": round(energy_low + energy_high + fee + vat, 2)}


@traced
def cycles(first_day: date, last_day) -> List[Dict]:
    """
    #Feature: Bill forecast
    #Scenario: 30-day cycles from the first day; the last one may run past the forecast
    #Complexity: Time O(c) - cycles

    Args:
        first_day: First day of cycle 1 (1 January of the forecast year).
        last_day: Last forecast day.

    Returns:
        One dict per cycle with cycle, start, end and full.

    Raises:
        ValueError: When an argument is None.
    """
    require("cycles", "first_day", first_day)
    require("cycles", "last_day", last_day)
    out, start, n = [], first_day, 1
    last_day = pd.Timestamp(last_day).date()
    while start <= last_day:
        end = start + timedelta(days=CYCLE_DAYS - 1)
        out.append({"cycle": n, "start": start, "end": end, "full": end <= last_day})
        start, n = end + timedelta(days=1), n + 1
    return out
