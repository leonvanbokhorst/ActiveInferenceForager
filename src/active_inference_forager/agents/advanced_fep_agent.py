import numpy as np
from typing import Dict
import logging
from pydantic import BaseModel, Field, model_validator, ConfigDict
from active_inference_forager.agents.belief_node import BeliefNode
from active_inference_forager.utils.numpy_fields import NumpyArrayField

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedFEPAgent(BaseModel):
    """
    Advanced agent operating under the Free Energy Principle and Active Inference framework.
    """

    root_belief: BeliefNode = Field(...)
    free_energy: float = Field(default=0.0)
    action_space: NumpyArrayField = Field(...)
    state_dim: int = Field(...)
    learning_rate: float = Field(default=0.1)
    discount_factor: float = Field(default=0.95)
    exploration_factor: float = Field(default=0.1)
    max_depth: int = Field(default=3)
    epsilon: float = Field(default=1e-6)  # Small constant for numerical stability

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def initialize_beliefs(self):
        """Initialize hierarchical belief structure."""
        if not self.root_belief.children:
            self._build_belief_hierarchy(self.root_belief, 0)
        return self

    def _build_belief_hierarchy(self, node: BeliefNode, level: int):
        """Recursively build the belief hierarchy."""
        if level >= self.max_depth:
            return
        for i in range(self.action_space.shape[0]):
            child = BeliefNode(
                mean=np.zeros(self.state_dim),
                precision=np.eye(self.state_dim)
                * (1.0 / (level + 1)),  # Decrease precision with depth
                level=level + 1,
            )
            node.children[f"action_{i}"] = child
            self._build_belief_hierarchy(child, level + 1)

    def update_belief(self, observation: np.ndarray) -> None:
        """Update beliefs based on new observation."""
        self._update_belief_recursive(self.root_belief, observation)
        self._update_free_energy()

    def _update_belief_recursive(self, node: BeliefNode, observation: np.ndarray):
        """Recursively update beliefs in the hierarchy."""
        node.update(observation, self.learning_rate)
        for child in node.children.values():
            self._update_belief_recursive(child, observation)

    def _update_free_energy(self) -> None:
        """Calculate and update the current free energy."""
        self.free_energy = self._calculate_free_energy_recursive(self.root_belief)

    def _calculate_free_energy_recursive(self, node: BeliefNode) -> float:
        """Recursively calculate free energy for the belief hierarchy."""
        kl_divergence = self._kl_divergence(node)
        for child in node.children.values():
            kl_divergence += self._calculate_free_energy_recursive(child)
        return kl_divergence

    def _kl_divergence(self, node: BeliefNode) -> float:
        """Calculate KL divergence for a belief node against a prior."""
        prior_mean = np.zeros_like(node.mean)
        prior_precision = np.eye(node.dim)

        # Add small constant to diagonal for numerical stability
        node_precision = node.precision + np.eye(node.dim) * self.epsilon

        try:
            # Use pseudo-inverse for more stable calculations
            node_cov = np.linalg.pinv(node_precision)
            prior_cov = np.linalg.pinv(prior_precision)

            kl = 0.5 * (
                np.trace(prior_precision @ node_cov)
                + (prior_mean - node.mean).T
                @ prior_precision
                @ (prior_mean - node.mean)
                - node.dim
                + np.log(max(np.linalg.det(node_precision), self.epsilon))
                - np.log(max(np.linalg.det(prior_precision), self.epsilon))
            )
        except np.linalg.LinAlgError:
            # If calculation fails, return a large KL divergence
            return 1e6

        return float(max(kl, 0))  # Ensure non-negative KL divergence

    def take_action(self, state: np.ndarray) -> np.ndarray:
        q_values = self._calculate_q_values(state)
        logger.debug(f"Q-values: {q_values}")
        if np.random.random() < self.exploration_factor:
            action = self.action_space[np.random.choice(len(self.action_space))]
        else:
            # Softmax action selection
            exp_q_values = np.exp(
                q_values - np.max(q_values)
            )  # Subtract max for numerical stability
            probabilities = exp_q_values / np.sum(exp_q_values)
            action_index = np.random.choice(len(self.action_space), p=probabilities)
            action = self.action_space[action_index]
        logger.debug(f"Chosen action: {action}")
        return action

    def _calculate_q_values(self, state: np.ndarray) -> np.ndarray:
        """Calculate Q-values for each action."""
        q_values = np.zeros(len(self.action_space))
        for i, action in enumerate(self.action_space):
            q_values[i] = -self._calculate_expected_free_energy(state, action)
        return q_values

    def _calculate_expected_free_energy(
        self, state: np.ndarray, action: np.ndarray, depth: int = 0
    ) -> float:
        if depth >= self.max_depth:
            return 0

        next_state = state + action

        immediate_fe = self._kl_divergence(self.root_belief)
        info_gain = self._calculate_information_gain(next_state)

        # Add an immediate reward estimate
        distance_to_origin = np.linalg.norm(next_state)
        immediate_reward = -distance_to_origin * 0.1

        future_fe = 0
        for next_action in self.action_space:
            future_fe += self._calculate_expected_free_energy(
                next_state, next_action, depth + 1
            )
        future_fe /= len(self.action_space)

        logger.debug(
            f"EFE: Immediate FE={immediate_fe}, Info Gain={info_gain}, Immediate Reward={immediate_reward}, Future FE={future_fe}"
        )
        return (
            immediate_fe
            + info_gain
            - immediate_reward
            + self.discount_factor * future_fe
        )

    def _calculate_information_gain(self, state: np.ndarray) -> float:
        """Calculate expected information gain for a given state."""
        prior_entropy = self._entropy(self.root_belief)
        posterior = BeliefNode(
            mean=self.root_belief.mean.copy(),
            precision=self.root_belief.precision.copy(),
        )
        posterior.update(state, self.learning_rate)
        posterior_entropy = self._entropy(posterior)
        return max(
            prior_entropy - posterior_entropy, 0
        )  # Ensure non-negative information gain

    def _entropy(self, node: BeliefNode) -> float:
        """Calculate entropy of a belief node."""
        try:
            # Use pseudo-inverse for more stable calculations
            cov = np.linalg.pinv(node.precision + np.eye(node.dim) * self.epsilon)
            return 0.5 * (
                node.dim * (1 + np.log(2 * np.pi))
                + np.log(max(np.linalg.det(cov), self.epsilon))
            )
        except np.linalg.LinAlgError:
            # If calculation fails, return a large entropy
            return 1e6

    def learn(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        reward: float,
    ):
        """Learn from experience using a form of model-based reinforcement learning."""
        logger.debug(
            f"Learning from: State={state}, Action={action}, Next State={next_state}, Reward={reward}"
        )
        # Update transition model (simplified)
        transition_error = next_state - (state + action)
        self.root_belief.update(transition_error, self.learning_rate)

        # Update beliefs
        self.update_belief(next_state)

    def reset(self) -> None:
        """Reset the agent's beliefs and free energy."""
        self.root_belief = BeliefNode(
            mean=np.zeros(self.state_dim), precision=np.eye(self.state_dim)
        )
        self.initialize_beliefs()
        self.free_energy = 0.0
