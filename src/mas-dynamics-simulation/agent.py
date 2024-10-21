# File: agent.py
from abc import ABC, abstractmethod, abstractproperty
from typing import Dict, Any


class Agent(ABC):
    """
    Represents an autonomous agent in the multi-agent system.
    Responsible for perceiving the environment, making decisions, and taking actions.
    """

    @abstractproperty
    def name(self) -> str:
        pass

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

    def __str__(self) -> str:
        return f"Agent({self.__class__.__name__})"


class Action(ABC):
    """
    Represents an action that can be taken by an agent.
    Defines how the action is executed in the environment.
    """

    @abstractproperty
    def name(self) -> str:
        pass

    @abstractmethod
    def execute(self, agent: Agent, environment: "Environment"):
        pass

    def __str__(self) -> str:
        return
