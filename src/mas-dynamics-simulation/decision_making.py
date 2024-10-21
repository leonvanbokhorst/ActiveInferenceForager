from abc import ABC, abstractmethod
from typing import List, Dict, Any


class DecisionEngine(ABC):
    """
    Responsible for agent decision-making processes.
    Evaluates options and chooses the best action for an agent.
    """

    @abstractmethod
    def evaluate_options(
        self, agent: "Agent", options: List["Action"], environment: "Environment"
    ) -> "Action":
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
        pass

    @abstractmethod
    def predict_others_actions(
        self, agent: "Agent", other_agents: List["Agent"], environment: "Environment"
    ) -> Dict["Agent", "Action"]:
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
        pass

    @abstractmethod
    def exert_influence(
        self, agent: "Agent", target: "Agent", environment: "Environment"
    ) -> float:
        pass
