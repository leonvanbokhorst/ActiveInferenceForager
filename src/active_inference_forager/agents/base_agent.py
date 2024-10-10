from pydantic import BaseModel, Field
import numpy as np
from active_inference_forager.agents.belief_node import BeliefNode
from active_inference_forager.utils.numpy_fields import NumpyArrayField


class BaseAgent(BaseModel):
    state_dim: int = Field(...)
    action_dim: int = Field(...)
    root_belief: BeliefNode = Field(default=None)
    action_space: NumpyArrayField = Field(default=None)
    learning_rate: float = Field(default=0.0001)
    discount_factor: float = Field(default=0.95)
    exploration_rate: float = Field(default=0.1)

    def initialize_belief_and_action_space(self):
        if self.root_belief is None:
            self.root_belief = BeliefNode(
                mean=np.zeros(self.state_dim),
                precision=np.eye(self.state_dim) * 0.1,
            )

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

    def take_action(self, state: np.ndarray) -> np.ndarray:
        q_values = self._calculate_q_values(state)
        if np.random.random() < self.exploration_rate:
            return self.action_space[np.random.choice(len(self.action_space))]
        else:
            return self.action_space[np.argmax(q_values)]

    def _calculate_q_values(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError("This method should be implemented by subclasses.")

    def learn(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray, reward: float):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def update_belief(self, observation: np.ndarray) -> None:
        self._update_belief_recursive(self.root_belief, observation)

    def _update_belief_recursive(self, node: BeliefNode, observation: np.ndarray):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def decay_exploration(self):
        self.exploration_rate *= 0.995
