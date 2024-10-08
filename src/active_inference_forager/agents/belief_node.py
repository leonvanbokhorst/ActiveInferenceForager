import numpy as np
from typing import Dict
from pydantic import BaseModel, Field, ConfigDict
from active_inference_forager.utils.numpy_fields import NumpyArrayField


class BeliefNode(BaseModel):
    """
    A class representing a belief node in a hierarchical model.

    Attributes:
    - mean (NumpyArrayField): The mean vector of the belief.
    - precision (NumpyArrayField): The precision matrix of the belief.
    - children (Dict[str, BeliefNode]): A dictionary of child belief nodes.
    - level (int): The level of the node in the hierarchy.
    - epsilon (float): A small constant to ensure numerical stability.
    - max_precision (float): Maximum allowable precision value.
    - max_mean (float): Maximum allowable mean value.
    """

    mean: NumpyArrayField = Field(...)
    precision: NumpyArrayField = Field(...)
    children: Dict[str, "BeliefNode"] = Field(default_factory=dict)
    level: int = Field(default=0)
    epsilon: float = Field(default=1e-6)
    max_precision: float = Field(default=1e6)
    max_mean: float = Field(default=1e2)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def update(self, observation: np.ndarray, learning_rate: float):
        """
        Update the belief based on a new observation.

        Parameters:
        - observation (np.ndarray): The new observation vector.
        - learning_rate (float): The learning rate for updating the belief.
        """
        try:
            prediction_error = observation - self.mean

            # Normalize precision matrix
            self.precision = self.precision / (np.trace(self.precision) + self.epsilon)

            # Use pseudo-inverse for more stable calculations
            precision_inv = np.linalg.pinv(
                self.precision + np.eye(self.dim) * self.epsilon
            )

            # Update mean with clamping
            mean_update = learning_rate * precision_inv.dot(prediction_error)
            self.mean += np.clip(mean_update, -self.max_mean, self.max_mean)
            self.mean = np.clip(self.mean, -self.max_mean, self.max_mean)

            # Update precision
            precision_update = (
                np.outer(prediction_error, prediction_error) - precision_inv
            )
            self.precision += learning_rate * precision_update

            # Ensure precision matrix remains positive definite and bounded
            eigvals, eigvecs = np.linalg.eigh(self.precision)
            eigvals = np.clip(eigvals, self.epsilon, self.max_precision)
            self.precision = eigvecs @ np.diag(eigvals) @ eigvecs.T
        except np.linalg.LinAlgError as e:
            print(f"An error occurred during matrix operations: {e}")

    @property
    def dim(self):
        """
        Return the dimension of the mean vector.

        Returns:
        - int: The dimension of the mean vector.
        """
        return self.mean.shape[0]
