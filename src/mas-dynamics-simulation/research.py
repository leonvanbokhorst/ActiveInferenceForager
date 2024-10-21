from abc import ABC, abstractmethod
from typing import List, Dict, Any


class ActionResearch(ABC):
    """
    Represents the action research process within the wicked problem context.
    Manages the cycle of planning, acting, observing, and reflecting.
    """

    @abstractmethod
    def plan(self, context: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def act(self, plan: Dict[str, Any], environment: "Environment") -> Dict[str, Any]:
        pass

    @abstractmethod
    def observe(
        self, action_results: Dict[str, Any], environment: "Environment"
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def reflect(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def engage_stakeholders(
        self, stakeholders: List["Agent"], environment: "Environment"
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def synthesize_findings(
        self, cycle_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def adapt_approach(self, synthesis: Dict[str, Any]):
        pass

    @abstractmethod
    def conduct_research_cycle(
        self, environment: "Environment", stakeholders: List["Agent"]
    ) -> Dict[str, Any]:
        pass


class WickedProblemResearch(ABC):
    """
    Encapsulates the process of researching and addressing a wicked problem.
    Manages problem definition, research iterations, and approach adaptation.
    """

    @abstractmethod
    def define_problem(self, description: str, stakeholders: List["Agent"]):
        pass

    @abstractmethod
    def conduct_iteration(self, environment: "Environment") -> Dict[str, Any]:
        pass

    @abstractmethod
    def analyze_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def adapt_approach(self, analysis: Dict[str, Any]):
        pass
