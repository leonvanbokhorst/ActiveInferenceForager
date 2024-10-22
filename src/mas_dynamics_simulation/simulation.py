from abc import ABC, abstractmethod
from typing import List, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import Agent
    from .environment import Environment
    from .research import WickedProblemResearch


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
        """
        Initializes the simulation with the given environment, agents, and wicked problem.

        Args:
            environment: The environment to initialize the simulation in.
            agents: The agents to initialize the simulation with.
            wicked_problem: The wicked problem to initialize the simulation with.
        """
        pass

    @abstractmethod
    def run_iteration(self) -> Dict[str, Any]:
        """
        Runs an iteration of the simulation.

        Returns:
            Dict[str, Any]: The results of the iteration.
        """
        pass

    @abstractmethod
    def generate_insights(self) -> List[str]:
        """
        Generates insights from the simulation results.

        Returns:
            List[str]: The insights from the simulation.
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        """
        Returns a string representation of the UnifiedSimulation.

        This method should be implemented by subclasses to provide a meaningful
        string representation of the UnifiedSimulation's current state.

        Returns:
            str: A string representation of the UnifiedSimulation.
        """
        pass
