"""Saudi calendar literals. Islamic dates are derived from the Umm al-Qura calendar (hijridate)."""
from typing import Dict, Tuple

# Saudi weekend is Friday and Saturday (pandas: Monday=0 ... Friday=4, Saturday=5).
WEEKEND_DAYS: Tuple[int, ...] = (4, 5)
FIXED_HOLIDAYS: Dict[Tuple[int, int], str] = {(2, 22): "founding_day", (9, 23): "national_day"}

# Hijri months: 9 = Ramadan, 10 = Shawwal, 12 = Dhul Hijjah.
RAMADAN_MONTH: int = 9
SHAWWAL_MONTH: int = 10
DHUL_HIJJAH_MONTH: int = 12
EID_AL_FITR_HOLIDAY_DAYS: int = 4      # 1-4 Shawwal
EID_AL_ADHA_FIRST_DAY: int = 9         # 9 Dhul Hijjah (Arafat) ...
EID_AL_ADHA_HOLIDAY_DAYS: int = 5      # ... through 13 Dhul Hijjah

CLIP_RAMADAN_DAYS: int = 60
CLIP_EID_ADHA_DAYS: int = 30
