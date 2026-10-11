"""Interface for running the whole pipeline."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Dict, Optional


class IPipelineService(ABC):
    """
    #Feature: Pipeline
    #Scenario: Callers (CLI, API) depend on this interface, never on the stages
    #Complexity: Time O(months * fit) - one model per month

    Data -> validation of the year -> forecast to year end -> dashboard.
    """

    @abstractmethod
    def run(self, refresh_sources: bool = False, log: Optional[Callable[[str], None]] = None):
        """
        #Feature: Pipeline
        #Scenario: Run every stage and save every result
        #Complexity: Time O(months * fit) - one model per month

        Args:
            refresh_sources: Re-read the warehouse and the Excel exports.
            log: Progress callback.

        Returns:
            A PipelineResults bundle.

        Raises:
            Errors.Misc.MeterMappingError: When an Excel export cannot be tied to one meter.
            Errors.Misc.HorizonExceeded: When the forecast needs more days than the model was trained for.
        """

    @abstractmethod
    def saved_results(self):
        """
        #Feature: Pipeline
        #Scenario: Reload the last full run without retraining
        #Complexity: Time O(n) - saved files

        Returns:
            A PipelineResults bundle.

        Raises:
            Errors.Misc.DataNotReady: When the pipeline has not run yet.
        """

    @abstractmethod
    def build_dashboard(self, results=None) -> Path:
        """
        #Feature: Pipeline
        #Scenario: Build the dashboard page from a run, or from the last saved run
        #Complexity: Time O(n) - page data

        Args:
            results: Optional PipelineResults; the saved run is used when omitted.

        Returns:
            Path of the HTML page.

        Raises:
            Errors.Misc.DataNotReady: When there is no saved run.
        """

    @abstractmethod
    def status(self) -> Dict[str, bool]:
        """
        #Feature: Pipeline
        #Scenario: Which saved results exist yet?
        #Complexity: Time O(k) - number of results checked

        Returns:
            result name -> whether its file exists.

        Raises:
            None.
        """
