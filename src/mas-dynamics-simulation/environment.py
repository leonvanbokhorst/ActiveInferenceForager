from abc import ABC, abstractmethod
from typing import Dict, Any


class Environment(ABC):
    """
    Represents the environment in which agents operate.
    Manages the state of the world and how agents interact with it.
    """

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def apply_action(self, agent: "Agent", action: "Action"):
        pass

    @abstractmethod
    def update(self):
        pass


class ComplexEnvironmentModel(Environment):
    """
    An advanced environment model that allows for dynamic variables and complex interactions.
    Suitable for representing intricate real-world scenarios.
    """

    @abstractmethod
    def add_variable(self, name: str, initial_value: float, update_function: callable):
        pass

    @abstractmethod
    def get_variable(self, name: str) -> float:
        pass

    @abstractmethod
    def set_variable(self, name: str, value: float):
        pass
