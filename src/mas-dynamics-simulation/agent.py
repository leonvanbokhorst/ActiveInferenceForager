# File: agent.py
from abc import ABC, abstractmethod
from typing import Dict, Any


class Agent(ABC):
    """
    Represents an autonomous agent in the multi-agent system.
    Responsible for perceiving the environment, making decisions, and taking actions.
    """

    @abstractmethod
    def perceive(self, environment: "Environment") -> Dict[str, Any]:
        pass

    @abstractmethod
    def decide(self, perception: Dict[str, Any]) -> "Action":
        pass

    @abstractmethod
    def act(self, action: "Action", environment: "Environment"):
        pass

    @abstractmethod
    def update(self, feedback: Dict[str, Any]):
        pass


class Action(ABC):
    """
    Represents an action that can be taken by an agent.
    Defines how the action is executed in the environment.
    """

    @abstractmethod
    def execute(self, agent: Agent, environment: "Environment"):
        pass
