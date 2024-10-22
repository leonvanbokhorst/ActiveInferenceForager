from abc import ABC, abstractmethod
from typing import Dict, Any, TYPE_CHECKING, List
import numpy as np

if TYPE_CHECKING:
    from .environment import Environment
    from .personality import Personality
    from .decision_making import DecisionEngine


class Agent(ABC):
    """
    Represents an autonomous agent in the multi-agent system. 
    Manages the agent's state, personality, expertise, and decision-making process.
    """

    def __init__(self, name: str, personality: "Personality", expertise: List[str], decision_engine: "DecisionEngine"):
        """
        Initializes the agent with the given name, personality, expertise, and decision-making engine.

        Args:
            name: The name of the agent.
            personality: The personality of the agent.
            expertise: The expertise of the agent.
            decision_engine: The decision-making engine of the agent.
        """
        self._name = name
        self._personality = personality
        self._expertise = expertise
        self._decision_engine = decision_engine

    @property
    def name(self) -> str:
        return self._name

    @property
    def personality(self) -> "Personality":
        return self._personality

    @property
    def expertise(self) -> List[str]:
        return self._expertise

    @property
    def decision_engine(self) -> "DecisionEngine":
        return self._decision_engine

    @abstractmethod
    def perceive(self, environment: "Environment") -> Dict[str, Any]:
        """
        Perceives the environment and returns a dictionary of perceptions.

        Args:
            environment: The environment to perceive.

        Returns:
            Dict[str, Any]: A dictionary of perceptions.
        """
        pass

    @abstractmethod
    def decide(self, perception: Dict[str, Any]) -> "Action":
        """
        Decides on an action based on the given perceptions.

        Args:
            perception: The perceptions to decide on.

        Returns:
            Action: The decided action.
        """
        pass

    @abstractmethod
    def act(self, action: "Action", environment: "Environment"):
        """
        Acts on the environment with the given action.

        Args:
            action: The action to act on.
            environment: The environment to act on.
        """
        pass

    @abstractmethod
    def update(self, feedback: Dict[str, Any]):
        """
        Updates the agent's state based on the given feedback.

        Args:
            feedback: The feedback to update the agent's state.
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        """
        Returns a string representation of the Agent.

        This method provides a meaningful string representation of the Agent's current state.

        Returns:
            str: A string representation of the Agent.
        """
        pass


class Action(ABC):
    """
    Represents an action that can be taken by an agent.
    Defines how the action is executed in the environment.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Returns the name of the action.

        Returns:
            str: The name of the action.
        """
        pass

    @abstractmethod
    def execute(self, agent: Agent, environment: "Environment"):
        """
        Executes the action in the environment.

        Args:
            agent: The agent executing the action.
            environment: The environment to execute the action in.
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        """
        Returns a string representation of the Action.

        This method should be implemented by subclasses to provide a meaningful
        string representation of the Action's current state.

        Returns:
            str: A string representation of the Action.
        """
        pass
