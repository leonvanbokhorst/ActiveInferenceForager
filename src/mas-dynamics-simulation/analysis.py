from abc import ABC, abstractmethod
from typing import List, Dict, Any


class EmergentBehaviorAnalyzer(ABC):
    """
    Analyzes the system for emergent behaviors and patterns.
    Detects and classifies complex behaviors arising from agent interactions.
    """

    @abstractmethod
    def detect_patterns(
        self, environment: "Environment", agents: List["Agent"]
    ) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def classify_behavior(self, pattern: Dict[str, Any]) -> str:
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
        pass

    @abstractmethod
    def generate_report(self, metrics: Dict[str, float]) -> str:
        pass
