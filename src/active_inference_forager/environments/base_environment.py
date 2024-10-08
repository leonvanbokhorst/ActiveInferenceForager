import numpy as np
from typing import Tuple
from pydantic import BaseModel, Field, ConfigDict
from active_inference_forager.utils.numpy_fields import NumpyArrayField


class BaseEnvironment(BaseModel):
    """Abstract base class for environments.

    Attributes:
        state_dim (int): Dimension of the state space.
        action_dim (int): Dimension of the action space.
        state (NumpyArrayField): Current state of the environment.
    """

    state_dim: int
    action_dim: int
    state: NumpyArrayField = Field(default_factory=lambda: np.array([0.0, 0.0]))

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """Take a step in the environment.

        Args:
            action (np.ndarray): The action to take.

        Returns:
            Tuple[np.ndarray, float, bool]: A tuple containing the new state,
            the reward obtained, and a boolean indicating if the episode is done.
        """
        raise NotImplementedError("The step method must be implemented by subclasses.")

    def reset(self) -> np.ndarray:
        """Reset the environment to an initial state.

        Returns:
            np.ndarray: The initial state of the environment.
        """
        raise NotImplementedError("The reset method must be implemented by subclasses.")
