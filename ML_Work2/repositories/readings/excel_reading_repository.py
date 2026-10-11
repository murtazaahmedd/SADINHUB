"""Telemetry exports delivered as .xlsx files."""
from pathlib import Path
from typing import List

import pandas as pd

from interfaces.repositories.readings.excel_reading_repository import IExcelReadingRepository
from utilities.guard import require, traced


class ExcelReadingRepository(IExcelReadingRepository):
    """
    #Feature: Excel exports
    #Scenario: Read export files from one folder
    #Complexity: Time O(f) - files

    Reads timestamp (epoch ms) and kWh columns from each export.
    """

    @traced
    def __init__(self, folder: Path) -> None:
        """
        #Feature: Excel exports
        #Scenario: Constructor injection of the export folder
        #Complexity: Time O(1) - constant

        Args:
            folder: Folder holding the .xlsx exports.

        Raises:
            ValueError: When folder is None.
        """
        require("ExcelReadingRepository.__init__", "folder", folder)
        self._folder = Path(folder)

    @traced
    def list_files(self) -> List[Path]:
        """
        #Feature: Excel exports
        #Scenario: Every export file, sorted by name
        #Complexity: Time O(f) - files

        Returns:
            Paths of the .xlsx files (empty when the folder is missing or empty).

        Raises:
            OSError: When the folder exists but cannot be read.
        """
        return sorted(self._folder.glob("*.xlsx")) if self._folder.exists() else []

    @traced
    def read_file(self, path: Path) -> pd.DataFrame:
        """
        #Feature: Excel exports
        #Scenario: One export as sorted (ts_utc, kwh) readings
        #Complexity: Time O(n log n) - rows, sorted

        Args:
            path: The export file.

        Returns:
            Columns ts_utc and kwh, sorted by time.

        Raises:
            ValueError: When path is None.
        """
        require("ExcelReadingRepository.read_file", "path", path)
        frame = pd.read_excel(path, usecols=["timestamp", "kWh"], dtype=str, engine="openpyxl").dropna()
        ts = pd.to_datetime(pd.to_numeric(frame["timestamp"], errors="coerce"), unit="ms")
        kwh = pd.to_numeric(frame["kWh"], errors="coerce")
        out = pd.DataFrame({"ts_utc": ts, "kwh": kwh}).dropna()
        return out.sort_values("ts_utc").reset_index(drop=True)
