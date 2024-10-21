from abc import ABC, abstractmethod
from typing import List, Dict, Any


class UnifiedSimulation(ABC):
    """
    Manages the overall simulation process.
    Coordinates agents, environment, and wicked problem research components.
    """

    @abstractmethod
    def initialize(
        self,
        environment: "Environment",
        agents: List["Agent"],
        wicked_problem: "WickedProblemResearch",
    ):
        pass

    @abstractmethod
    def run_iteration(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def generate_insights(self) -> List[str]:
        pass
