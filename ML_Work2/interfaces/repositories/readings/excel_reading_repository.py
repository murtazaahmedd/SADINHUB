"""Interface for telemetry Excel exports."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import pandas as pd


class IExcelReadingRepository(ABC):
    """
    #Feature: Excel exports
    #Scenario: Services read exports through this interface
    #Complexity: Time O(f) - files

    Telemetry exports delivered as .xlsx files.
    """

    @abstractmethod
    def list_files(self) -> List[Path]:
        """
        #Feature: Excel exports
        #Scenario: Every export file, sorted by name
        #Complexity: Time O(f) - files

        Returns:
            Paths of the .xlsx files (empty when there are none).

        Raises:
            Errors.HTTP.InternalServerError: When the folder cannot be read.
        """

    @abstractmethod
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
