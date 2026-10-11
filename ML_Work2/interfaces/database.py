"""Interface for anything that hands out a database connection."""
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager


class ISessionFactory(ABC):
    """
    #Feature: Database access
    #Scenario: Repositories receive connections through an interface, never create them
    #Complexity: Time O(1) - constant

    Hands out connections to the warehouse.
    """

    @abstractmethod
    def session(self) -> AbstractContextManager:
        """
        #Feature: Database access
        #Scenario: One connection for one unit of work
        #Complexity: Time O(1) - constant

        Returns:
            A context manager yielding a SQLAlchemy Connection.

        Raises:
            Errors.HTTP.InternalServerError: When the connection cannot be opened.
        """
