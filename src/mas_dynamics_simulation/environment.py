from abc import ABC, abstractmethod
from typing import Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import Agent
    from .agent import Action


class Environment(ABC):
    """
    Represents the environment in which agents operate.
    Manages the state of the world and how agents interact with it.
    """

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """
        Returns the current state of the environment.

        Returns:
            Dict[str, Any]: The current state of the environment.
        """
        pass

    @abstractmethod
    def apply_action(self, agent: "Agent", action: "Action"):
        """
        Applies the given action to the environment.

        Args:
            agent: The agent to apply the action to.
            action: The action to apply.
        """
        pass

    @abstractmethod
    def update(self):
        """
        Updates the environment state.
        """
        pass


class ComplexEnvironmentModel(Environment):
    """
    An advanced environment model that allows for dynamic variables and complex interactions.
    Suitable for representing intricate real-world scenarios.
    """

    @abstractmethod
    def add_variable(self, name: str, initial_value: float, update_function: callable):
        """
        Adds a new variable to the environment.

        Args:
            name: The name of the variable.
            initial_value: The initial value of the variable.
            update_function: The function to update the variable.
        """
        pass

    @abstractmethod
    def get_variable(self, name: str) -> float:
        """
        Gets the value of a variable from the environment.

        Args:
            name: The name of the variable.

        Returns:
            float: The value of the variable.
        """
        pass

    @abstractmethod
    def set_variable(self, name: str, value: float):
        """
        Sets the value of a variable in the environment.

        Args:
            name: The name of the variable.
            value: The value to set the variable to.
        """
        pass
