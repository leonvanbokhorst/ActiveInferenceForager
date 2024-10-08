import numpy as np
import math
from typing import Tuple, Optional
from pydantic import BaseModel, Field, ConfigDict
from active_inference_forager.utils.numpy_fields import NumpyArrayField
from active_inference_forager.environments.base_environment import BaseEnvironment


class SimpleEnvironment(BaseEnvironment):
    """
    A simple 2D environment where an agent can move in a continuous space.

    Attributes:
        max_steps (int): Maximum number of steps before the episode ends.
        state_dim (int): Dimensionality of the state space.
        action_dim (int): Dimensionality of the action space.
        steps (int): Current step count in the episode.
        goal_radius (float): Radius within which the goal is considered reached.
        previous_distance (Optional[float]): Distance to the goal in the previous step.
        action_space (NumpyArrayField): Array of possible actions.
    """

    max_steps: int = Field(default=100)
    state_dim: int = Field(default=2)
    action_dim: int = Field(default=2)
    steps: int = Field(default=0)
    goal_radius: float = Field(default=0.1)
    previous_distance: Optional[float] = Field(default=None)
    action_space: Optional[NumpyArrayField] = Field(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data):
        """
        Initialize the SimpleEnvironment.

        Args:
            **data: Arbitrary keyword arguments for environment configuration.
        """
        super().__init__(**data)
        self.state = np.zeros(self.state_dim)
        self.steps = 0
        if self.action_space is None:
            self.action_space = np.array(
                [
                    [-0.1, 0],
                    [0.1, 0],
                    [0, -0.1],
                    [0, 0.1],
                    [-0.1, -0.1],
                    [-0.1, 0.1],
                    [0.1, -0.1],
                    [0.1, 0.1],
                ]
            )
        self.reset()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """
        Take a step in the environment using the given action.

        Args:
            action (np.ndarray): The action to take.

        Returns:
            Tuple[np.ndarray, float, bool]: The new state, reward, and done flag.
        """
        old_distance = np.linalg.norm(self.state)
        self.state = self.state + action
        new_distance = np.linalg.norm(self.state)

        reward = self._calculate_reward(old_distance, new_distance)

        self.steps += 1
        done = new_distance < self.goal_radius or self.steps >= self.max_steps
        return self.state.copy(), reward, done

    def reset(self) -> np.ndarray:
        """
        Reset the environment to an initial state.

        Returns:
            np.ndarray: The initial state.
        """
        self.state = np.random.uniform(-1, 1, self.state_dim)
        self.steps = 0
        self.previous_distance = np.linalg.norm(self.state)
        return self.state.copy()

    def _calculate_reward(self, old_distance: float, new_distance: float) -> float:
        """
        Calculate the reward based on the change in distance to the goal.

        Args:
            old_distance (float): Distance to the goal before the action.
            new_distance (float): Distance to the goal after the action.

        Returns:
            float: The calculated reward.
        """
        # Reward for moving towards the goal
        distance_reward = (old_distance - new_distance) * 10.0  # Increased weight
        # Penalty for being far from the goal
        distance_penalty = -new_distance * 0.1
        # Reward for reaching the goal
        goal_reward = 50 if new_distance < self.goal_radius else 0
        # Small penalty for each step to encourage efficiency
        step_penalty = -0.1

        return distance_reward + distance_penalty + goal_reward + step_penalty
