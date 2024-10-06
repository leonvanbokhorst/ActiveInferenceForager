import numpy as np
import math
from typing import Tuple
from pydantic import BaseModel, Field, ConfigDict
from active_inference_forager.utils.numpy_fields import NumpyArrayField


class Environment(BaseModel):
    """Abstract base class for environments."""

    state_dim: int
    action_dim: int
    state: NumpyArrayField = Field(default_factory=lambda: np.array([0.0, 0.0]))

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """Take a step in the environment."""
        raise NotImplementedError

    def reset(self) -> np.ndarray:
        """Reset the environment."""
        raise NotImplementedError


import numpy as np
from typing import Tuple
from pydantic import Field, ConfigDict
from active_inference_forager.utils.numpy_fields import NumpyArrayField
from active_inference_forager.agents.environment import Environment


class SimpleEnvironment(Environment):
    max_steps: int = Field(default=100)
    state_dim: int = Field(default=2)
    action_dim: int = Field(default=2)
    steps: int = Field(default=0)
    goal_radius: float = Field(default=0.1)
    previous_distance: float = Field(default=None)
    action_space: NumpyArrayField = Field(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data):
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
        old_distance = np.linalg.norm(self.state)
        self.state = self.state + action
        new_distance = np.linalg.norm(self.state)

        # Reward for moving towards the goal

        distance_reward = old_distance - new_distance * 10.0  # Increased weight
        # Penalty for being far from the goal
        distance_penalty = -new_distance * 0.1

        # Reward for reaching the goal
        goal_reward = 50 if new_distance < self.goal_radius else 0

        # Small penalty for each step to encourage efficiency
        step_penalty = -0.1

        reward = distance_reward + distance_penalty + goal_reward + step_penalty

        self.steps += 1
        done = new_distance < self.goal_radius or self.steps >= self.max_steps
        return self.state.copy(), reward, done

    def reset(self) -> np.ndarray:
        self.state = np.random.uniform(-1, 1, self.state_dim)
        self.steps = 0
        self.previous_distance = np.linalg.norm(self.state)
        return self.state.copy()
