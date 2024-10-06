import numpy as np
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
