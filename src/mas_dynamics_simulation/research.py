from abc import ABC, abstractmethod
from typing import List, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import Agent
    from .environment import Environment


class ActionResearch(ABC):
    """
    Represents the action research process within the wicked problem context.
    Manages the cycle of planning, acting, observing, and reflecting.
    """

    @abstractmethod
    def plan(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Plans an action based on the given context.

        Args:
            context: The context to plan the action in.

        Returns:
            Dict[str, Any]: The planned action.
        """
        pass

    @abstractmethod
    def act(self, plan: Dict[str, Any], environment: "Environment") -> Dict[str, Any]:
        """
        Acts on the environment based on the given plan.
        
        Args:
            plan: The plan to act on.
            environment: The environment to act on.

        Returns:
            Dict[str, Any]: The action results.
        """
        pass

    @abstractmethod
    def observe(
        self, action_results: Dict[str, Any], environment: "Environment"
    ) -> Dict[str, Any]:
        """
        Observes the results of the action.

        Args:
            action_results: The results of the action.
            environment: The environment to observe.

        Returns:
            Dict[str, Any]: The observations.
        """
        pass

    @abstractmethod
    def reflect(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reflects on the observations.

        Args:
            observations: The observations to reflect on.

        Returns:
            Dict[str, Any]: The reflections.
        """
        pass

    @abstractmethod
    def engage_stakeholders(
        self, stakeholders: List["Agent"], environment: "Environment"
    ) -> Dict[str, Any]:
        """
        Engages stakeholders in the research process.

        Args:
            stakeholders: The stakeholders to engage.
            environment: The environment to engage stakeholders in.

        Returns:
            Dict[str, Any]: The engagement results.
        """
        pass

    @abstractmethod
    def synthesize_findings(
        self, cycle_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Synthesizes the findings from the research cycle.

        Args:
            cycle_results: The results from the research cycle.

        Returns:
            Dict[str, Any]: The synthesized findings.
        """
        pass

    @abstractmethod
    def adapt_approach(self, synthesis: Dict[str, Any]):
        """
        Adapts the approach based on the synthesized findings.

        Args:
            synthesis: The synthesized findings.
        """
        pass

    @abstractmethod
    def conduct_research_cycle(
        self, environment: "Environment", stakeholders: List["Agent"]
    ) -> Dict[str, Any]:
        """
        Conducts a research cycle.

        Args:
            environment: The environment to conduct the research cycle in.
            stakeholders: The stakeholders to engage in the research cycle.

        Returns:
            Dict[str, Any]: The results of the research cycle.
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        """
        Returns a string representation of the ActionResearch.

        This method should be implemented by subclasses to provide a meaningful
        string representation of the ActionResearch's current state.

        Returns:
            str: A string representation of the ActionResearch.
        """
        pass


class WickedProblemResearch(ABC):
    """
    Encapsulates the process of researching and addressing a wicked problem.
    Manages problem definition, research iterations, and approach adaptation.
    """

    @abstractmethod
    def define_problem(self, description: str, stakeholders: List["Agent"]):
        """
        Defines the wicked problem.

        Args:
            description: The description of the wicked problem.
            stakeholders: The stakeholders involved in the problem.
        """
        pass

    @abstractmethod
    def conduct_iteration(self, environment: "Environment") -> Dict[str, Any]:
        """
        Conducts an iteration of the research process.

        Args:
            environment: The environment to conduct the iteration in.

        Returns:
            Dict[str, Any]: The results of the iteration.
        """
        pass

    @abstractmethod
    def analyze_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes the results of the research process.

        Args:
            results: The results of the research process.

        Returns:
            Dict[str, Any]: The analysis of the results.
        """
        pass

    @abstractmethod
    def adapt_approach(self, analysis: Dict[str, Any]):
        """
        Adapts the approach based on the analysis of the results.

        Args:
            analysis: The analysis of the results.
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        """
        String representation of the WickedProblemResearch.

        This method should be implemented by subclasses to provide a meaningful
        string representation of the WickedProblemResearch's current state.

        Returns:
            str: A string representation of the WickedProblemResearch.
        """
        pass
