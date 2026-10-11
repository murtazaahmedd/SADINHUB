"""Interface for building the single-file dashboard."""
from abc import ABC, abstractmethod
from pathlib import Path


class IDashboardService(ABC):
    """
    #Feature: Dashboard
    #Scenario: Callers depend on this interface, never on the page builder
    #Complexity: Time O(n) - page data

    Builds the offline HTML dashboard.
    """

    @abstractmethod
    def build(self, results) -> Path:
        """
        #Feature: Dashboard
        #Scenario: Assemble the page from a pipeline run
        #Complexity: Time O(n) - page data

        Args:
            results: A PipelineResults bundle.

        Returns:
            Path of the written HTML file.

        Raises:
            ValueError: When results is None.
        """
