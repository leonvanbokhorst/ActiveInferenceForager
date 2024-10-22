from abc import ABC, abstractmethod
from typing import List, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import Agent
    from .agent import Action
    from .environment import Environment


class DecisionEngine(ABC):
    """
    Responsible for agent decision-making processes.
    Evaluates options and chooses the best action for an agent.
    """

    @abstractmethod
    def evaluate_options(
        self, agent: "Agent", options: List["Action"], environment: "Environment"
    ) -> "Action":
        """
        Evaluates the options and chooses the best action for the agent.

        Args:
            agent: The agent to evaluate the options for.
            options: The options to evaluate.
            environment: The environment to evaluate the options in.

        Returns:
            Action: The best action for the agent.
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        """
        Returns a string representation of the DecisionEngine.

        This method should be implemented by subclasses to provide a meaningful
        string representation of the DecisionEngine's current state.

        Returns:
            str: A string representation of the DecisionEngine.
        """
        pass


class GameTheoryModule(ABC):
    """
    Applies game theory concepts to agent interactions.
    Used to calculate payoffs and predict actions in multi-agent scenarios.
    """

    @abstractmethod
    def calculate_payoff(
        self, agent: "Agent", action: "Action", environment: "Environment"
    ) -> float:
        """
        Calculates the payoff for the given action.

        Args:
            agent: The agent to calculate the payoff for.
            action: The action to calculate the payoff for.
            environment: The environment to calculate the payoff in.

        Returns:
            float: The payoff for the given action.
        """
        pass

    @abstractmethod
    def predict_others_actions(
        self, agent: "Agent", other_agents: List["Agent"], environment: "Environment"
    ) -> Dict["Agent", "Action"]:
        """
        Predicts the actions of other agents.

        Args:
            agent: The agent to predict the actions of other agents for.
            other_agents: The other agents to predict the actions of.
            environment: The environment to predict the actions in.

        Returns:
            Dict["Agent", "Action"]: The predicted actions of other agents.
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        """
        Returns a string representation of the GameTheoryModule.

        This method should be implemented by subclasses to provide a meaningful
        string representation of the GameTheoryModule's current state.

        Returns:
            str: A string representation of the GameTheoryModule.
        """
        pass


class PoliticalBehaviorModule(ABC):
    """
    Handles political interactions between agents.
    Manages coalition formation and influence exertion.
    """

    @abstractmethod
    def form_coalition(
        self, agent: "Agent", other_agents: List["Agent"], environment: "Environment"
    ) -> List["Agent"]:
        """
        Forms a coalition of agents.

        Args:
            agent: The agent to form the coalition with.
            other_agents: The other agents to form the coalition with.
            environment: The environment to form the coalition in.

        Returns:
            List["Agent"]: The coalition of agents.
        """
        pass

    @abstractmethod
    def exert_influence(
        self, agent: "Agent", target: "Agent", environment: "Environment"
    ) -> float:
        """
        Exerts influence on the target agent.

        Args:
            agent: The agent to exert influence on.
            target: The target agent to exert influence on.
            environment: The environment to exert influence in.

        Returns:
            float: The influence exerted on the target agent.
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        """
        Returns a string representation of the PoliticalBehaviorModule.

        This method should be implemented by subclasses to provide a meaningful
        string representation of the PoliticalBehaviorModule's current state.

        Returns:
            str: A string representation of the PoliticalBehaviorModule.
        """
        pass
