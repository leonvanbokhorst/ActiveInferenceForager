from abc import ABC, abstractmethod
from typing import List, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import Agent
    from .environment import Environment


class EmergentBehaviorAnalyzer(ABC):
    """
    Analyzes the system for emergent behaviors and patterns.
    Detects and classifies complex behaviors arising from agent interactions.
    """

    @abstractmethod
    def detect_patterns(
        self, environment: "Environment", agents: List["Agent"]
    ) -> List[Dict[str, Any]]:
        """
        Detects patterns in the environment and agents.

        Args:
            environment: The environment to detect patterns in.
            agents: The agents to detect patterns in.

        Returns:
            List[Dict[str, Any]]: The detected patterns.
        """
        pass

    @abstractmethod
    def classify_behavior(self, pattern: Dict[str, Any]) -> str:
        """
        Classifies the behavior of the agents.

        Args:
            pattern: The pattern to classify.

        Returns:
            str: The classified behavior.
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        """
        Returns a string representation of the EmergentBehaviorAnalyzer.

        This method should be implemented by subclasses to provide a meaningful
        string representation of the EmergentBehaviorAnalyzer's current state.

        Returns:
            str: A string representation of the EmergentBehaviorAnalyzer.
        """
        pass


class SystemWideAnalyzer(ABC):
    """
    Performs system-wide analysis of the multi-agent environment.
    Calculates metrics and generates reports on overall system behavior.
    """

    @abstractmethod
    def calculate_metrics(
        self, environment: "Environment", agents: List["Agent"]
    ) -> Dict[str, float]:
        """
        Calculates metrics from the environment and agents.

        Returns:
            Dict[str, float]: The calculated metrics.
        """
        pass

    @abstractmethod
    def generate_report(self, metrics: Dict[str, float]) -> str:
        """
        Generates a report from the calculated metrics.

        Args:
            metrics: The metrics to generate a report from.

        Returns:
            str: The generated report.
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        """
        Returns a string representation of the SystemWideAnalyzer.

        This method should be implemented by subclasses to provide a meaningful
        string representation of the SystemWideAnalyzer's current state.

        Returns:
            str: A string representation of the SystemWideAnalyzer.
        """
        pass


class Environment(ABC):
    """
    Represents the environment in which agents interact.
    """

    @abstractmethod
    def __str__(self) -> str:
        """
        Returns a string representation of the Environment.

        This method should be implemented by subclasses to provide a meaningful
        string representation of the Environment's current state.

        Returns:
            str: A string representation of the Environment.
        """
        pass
