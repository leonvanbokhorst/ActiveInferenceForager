import numpy as np
from typing import Dict
from pydantic import BaseModel, Field, ConfigDict
from active_inference_forager.utils.numpy_fields import NumpyArrayField


class BeliefNode(BaseModel):
    """Represents a node in the hierarchical belief structure."""

    mean: NumpyArrayField = Field(...)
    precision: NumpyArrayField = Field(...)
    children: Dict[str, "BeliefNode"] = Field(default_factory=dict)
    level: int = Field(default=0)
    epsilon: float = Field(default=1e-6)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def update(self, observation: np.ndarray, learning_rate: float):
        """Update belief based on new observation."""
        prediction_error = observation - self.mean

        # Use pseudo-inverse for more stable calculations
        precision_inv = np.linalg.pinv(self.precision + np.eye(self.dim) * self.epsilon)

        self.mean += learning_rate * precision_inv.dot(prediction_error)
        self.precision += learning_rate * (
            np.outer(prediction_error, prediction_error) - precision_inv
        )

        # Ensure precision matrix remains positive definite
        eigvals, eigvecs = np.linalg.eigh(self.precision)
        eigvals = np.maximum(eigvals, self.epsilon)
        self.precision = eigvecs @ np.diag(eigvals) @ eigvecs.T

    @property
    def dim(self):
        return self.mean.shape[0]
