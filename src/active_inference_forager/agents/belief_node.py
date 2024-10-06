import numpy as np
from typing import Dict
from pydantic import BaseModel, Field, ConfigDict
from active_inference_forager.utils.numpy_fields import NumpyArrayField


class BeliefNode(BaseModel):
    mean: NumpyArrayField = Field(...)
    precision: NumpyArrayField = Field(...)
    children: Dict[str, "BeliefNode"] = Field(default_factory=dict)
    level: int = Field(default=0)
    epsilon: float = Field(default=1e-6)
    max_precision: float = Field(default=1e6)
    max_mean: float = Field(default=1e2)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def update(self, observation: np.ndarray, learning_rate: float):
        """Update belief based on new observation."""
        prediction_error = observation - self.mean

        # Normalize precision matrix
        self.precision = self.precision / (np.trace(self.precision) + self.epsilon)

        # Use pseudo-inverse for more stable calculations
        precision_inv = np.linalg.pinv(self.precision + np.eye(self.dim) * self.epsilon)

        # Update mean with clamping
        mean_update = learning_rate * precision_inv.dot(prediction_error)
        self.mean += np.clip(mean_update, -self.max_mean, self.max_mean)
        self.mean = np.clip(self.mean, -self.max_mean, self.max_mean)

        # Update precision
        precision_update = np.outer(prediction_error, prediction_error) - precision_inv
        self.precision += learning_rate * precision_update

        # Ensure precision matrix remains positive definite and bounded
        eigvals, eigvecs = np.linalg.eigh(self.precision)
        eigvals = np.clip(eigvals, self.epsilon, self.max_precision)
        self.precision = eigvecs @ np.diag(eigvals) @ eigvecs.T

    @property
    def dim(self):
        return self.mean.shape[0]
