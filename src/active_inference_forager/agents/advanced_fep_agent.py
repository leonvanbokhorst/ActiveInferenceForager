import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import logging
from typing import Dict, List, Tuple
from pydantic import BaseModel, Field, model_validator, ConfigDict
from scipy.stats import entropy
from collections import deque

from active_inference_forager.agents.belief_node import BeliefNode
from active_inference_forager.utils.numpy_fields import NumpyArrayField


# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all log levels

# Create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# Create formatter
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")  # - %(name)s

# Add formatter to ch
ch.setFormatter(formatter)

# Add ch to logger
logger.addHandler(ch)

# # log to file
# fh = logging.FileHandler("active_inference_forager.log")
# fh.setLevel(logging.DEBUG)
# fh.setFormatter(formatter)
# logger.addHandler(fh)


class AdvancedFEPAgent(BaseModel):

    episode_rewards: List[float] = Field(default_factory=list)
    episode_lengths: List[int] = Field(default_factory=list)
    total_steps: int = Field(default=0)

    state_dim: int = Field(default=2)
    action_dim: int = Field(default=2)
    root_belief: BeliefNode = Field(default=None)
    action_space: NumpyArrayField = Field(default=None)

    learning_rate: float = Field(default=0.0001)
    max_kl: float = Field(default=10.0)  # Increased from 5.0
    max_fe: float = Field(default=100.0)  # Increased from 50.0
    epsilon: float = Field(default=1e-7)

    discount_factor: float = Field(default=0.95)
    initial_exploration_factor: float = Field(default=0.3)
    final_exploration_factor: float = Field(default=0.05)
    exploration_factor: float = Field(default=0.3)
    exploration_decay: float = Field(default=0.995)
    exploration_rate: float = Field(default=0.1)

    initial_mean: NumpyArrayField = Field(default=None)
    initial_precision: NumpyArrayField = Field(default=None)

    min_precision: float = Field(default=0.01)
    max_precision: float = Field(default=100.0)

    max_mean: float = Field(default=1e2)
    max_depth: int = Field(default=2)

    belief_regularization: float = Field(default=0.01)
    max_belief_value: float = Field(default=100.0)

    buffer_size: int = Field(default=1000)
    memory_buffer: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]] = Field(
        default_factory=list
    )

    free_energy: float = Field(default=0.0)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def initialize_belief_and_action_space(self):
        if self.initial_mean is None:
            self.initial_mean = np.zeros(self.state_dim)
        if self.initial_precision is None:
            self.initial_precision = np.eye(self.state_dim) * 0.1

        if self.root_belief is None:
            self.root_belief = BeliefNode(
                mean=self.initial_mean,
                precision=self.initial_precision,
                max_precision=self.max_precision,
                max_mean=self.max_mean,
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

        self.update_free_energy()
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
        self._regularize_beliefs()
        self._update_free_energy()

    def _update_belief_recursive(self, node: BeliefNode, observation: np.ndarray):
        """Recursively update beliefs in the hierarchy."""
        prediction_error = observation - node.mean
        node.precision += (
            np.outer(prediction_error, prediction_error) * self.learning_rate
        )
        node.precision = np.clip(node.precision, self.min_precision, self.max_precision)

        precision_inv = np.linalg.inv(node.precision + np.eye(node.dim) * self.epsilon)
        node.mean += precision_inv.dot(prediction_error) * self.learning_rate
        node.mean = np.clip(node.mean, -self.max_belief_value, self.max_belief_value)

        for child in node.children.values():
            self._update_belief_recursive(child, observation)

    def _regularize_beliefs(self):
        """Apply regularization to prevent belief values from growing too large."""
        self.root_belief.mean *= 1 - self.belief_regularization
        self.root_belief.precision += (
            np.eye(self.root_belief.dim) * self.belief_regularization
        )

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

        try:
            node_cov = np.linalg.inv(node.precision + np.eye(node.dim) * self.epsilon)
            prior_cov = np.linalg.inv(prior_precision + np.eye(node.dim) * self.epsilon)

            kl = 0.5 * (
                np.trace(prior_precision @ node_cov)
                + (prior_mean - node.mean).T
                @ prior_precision
                @ (prior_mean - node.mean)
                - node.dim
                + np.log(np.linalg.det(node.precision) / np.linalg.det(prior_precision))
            )
        except np.linalg.LinAlgError:
            return self.max_kl

        return float(min(max(kl, 0), self.max_kl))

    def _entropy(self, node: BeliefNode) -> float:
        """Calculate entropy of a belief node."""
        try:
            # Normalize precision matrix
            precision = node.precision / (np.trace(node.precision) + self.epsilon)
            # Use pseudo-inverse for more stable calculations
            cov = np.linalg.pinv(precision + np.eye(node.dim) * self.epsilon)
            return 0.5 * (
                node.dim * (1 + np.log(2 * np.pi))
                + np.log(max(np.linalg.det(cov), self.epsilon))
            )
        except np.linalg.LinAlgError:
            return self.max_fe  # Return maximum allowed free energy value

    def take_action(self, state: np.ndarray) -> np.ndarray:
        q_values = self._calculate_q_values(state)
        if np.random.random() < self.exploration_rate:
            return self.action_space[np.random.choice(len(self.action_space))]
        else:
            return self.action_space[np.argmax(q_values)]

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

        immediate_fe = min(self._kl_divergence(self.root_belief), self.max_fe)
        info_gain = self._calculate_information_gain(next_state)

        distance_to_origin = np.linalg.norm(next_state)
        immediate_reward = -distance_to_origin * 0.1

        future_fe = 0
        for next_action in self.action_space:
            future_fe += self._calculate_expected_free_energy(
                next_state, next_action, depth + 1
            )
        future_fe /= len(self.action_space)

        total_fe = (
            immediate_fe * 0.1  # Reduce the impact of immediate free energy
            + info_gain
            - immediate_reward
            + self.discount_factor * future_fe
        )

        return min(total_fe, self.max_fe)

    def _add_exploration_noise(self):
        """Add noise to beliefs to encourage exploration."""
        noise = np.random.normal(
            0, self.exploration_rate, size=self.root_belief.mean.shape
        )
        self.root_belief.mean += noise
        self.root_belief.precision = np.clip(
            self.root_belief.precision, self.min_precision, self.max_precision
        )

    def _get_distribution(self, node: BeliefNode) -> np.ndarray:
        """Get a discrete probability distribution from a belief node."""
        cov = np.linalg.inv(node.precision + np.eye(node.dim) * self.epsilon)
        samples = np.random.multivariate_normal(node.mean, cov, size=1000)
        hist, _ = np.histogramdd(samples, bins=10)
        return hist / np.sum(hist)

    def _calculate_information_gain(self, state: np.ndarray) -> float:
        """Calculate expected information gain for a given state."""
        prior_dist = self._get_distribution(self.root_belief)
        posterior = BeliefNode(
            mean=self.root_belief.mean.copy(),
            precision=self.root_belief.precision.copy(),
        )
        posterior.update(state, self.learning_rate)
        posterior_dist = self._get_distribution(posterior)

        kl_div = entropy(prior_dist.flatten(), posterior_dist.flatten())
        return min(kl_div, self.max_fe)

    def decay_exploration(self):
        self.exploration_factor = max(
            self.final_exploration_factor,
            self.exploration_factor * self.exploration_decay,
        )

    def add_to_memory(self, state, action, next_state, reward):
        if len(self.memory_buffer) >= self.buffer_size:
            self.memory_buffer.pop(0)
        self.memory_buffer.append((state, action, next_state, reward))

    def update_free_energy(self):
        """Update the current free energy of the agent."""
        self.free_energy = self._kl_divergence(self.root_belief)

    def learn(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        reward: float,
    ):
        # logger.debug(
        #     f"Learning from: State={state}, Action={action}, Next State={next_state}, Reward={reward}"
        # )
        self.add_to_memory(state, action, next_state, reward)

        # Update transition model
        transition_error = next_state - (state + action)
        self.root_belief.update(transition_error, self.learning_rate)

        # Update beliefs
        self.update_belief(next_state)

        # Update free energy
        self.update_free_energy()

        # Log belief statistics
        # logger.debug(
        #     f"Updated belief - Mean: {self.root_belief.mean}, Precision: {self.root_belief.precision}"
        # )
        # logger.debug(f"Current Free Energy: {self.free_energy}")

        # Decay exploration factor
        self.decay_exploration()

        self.total_steps += 1

        # Log performance metrics every 100 steps
        if self.total_steps % 100 == 0:
            avg_reward = (
                np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0
            )
            avg_length = (
                np.mean(self.episode_lengths[-10:]) if self.episode_lengths else 0
            )
            # logger.info(
            #     f"Steps: {self.total_steps}, "
            #     f"Avg Rew: {avg_reward:.2f}, "
            #     f"Avg Ep Len: {avg_length:.2f}"
            # )

    def learn_from_memory(self, batch_size=32):
        if len(self.memory_buffer) < batch_size:
            return
        batch = random.sample(self.memory_buffer, batch_size)
        for state, action, next_state, reward in batch:
            self._update_from_sample(state, action, next_state, reward)

    def _update_from_sample(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        reward: float,
    ):
        # Update transition model
        transition_error = next_state - (state + action)
        self.root_belief.update(transition_error, self.learning_rate)

        # Update beliefs
        self.update_belief(next_state)

    def train(self, batch_size=32):
        self.learn_from_memory(batch_size)

    def reset(self) -> None:
        """Reset the agent's beliefs and free energy."""
        self.root_belief = BeliefNode(
            mean=np.zeros(self.state_dim),
            precision=np.eye(self.state_dim) * 0.1,  # Start with some uncertainty
            max_precision=self.max_precision,
            max_mean=self.max_mean,
        )
        self._build_belief_hierarchy(self.root_belief, 0)
        self.free_energy = 0.0
        self.exploration_factor = self.initial_exploration_factor

        if self.episode_rewards:
            self.episode_rewards.append(0)
            self.episode_lengths.append(0)

    def _build_belief_hierarchy(self, node: BeliefNode, level: int):
        """Recursively build the belief hierarchy."""
        if level >= self.max_depth:
            return
        for i in range(self.action_space.shape[0]):
            child = BeliefNode(
                mean=np.zeros(self.state_dim),
                precision=np.eye(self.state_dim)
                * (0.1 / (level + 1)),  # Decrease precision with depth
                level=level + 1,
                max_precision=self.max_precision,
                max_mean=self.max_mean,
            )
            node.children[f"action_{i}"] = child
            self._build_belief_hierarchy(child, level + 1)


class ImprovedAdvancedFEPAgent(AdvancedFEPAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _calculate_expected_free_energy(
        self, state: np.ndarray, action: np.ndarray, depth: int = 0
    ) -> float:
        if depth >= self.max_depth:
            return 0

        next_state = state + action

        immediate_fe = min(self._kl_divergence(self.root_belief), self.max_fe)
        info_gain = self._calculate_information_gain(next_state)

        distance_to_origin = np.linalg.norm(next_state)
        immediate_reward = -distance_to_origin * 0.1

        future_fe = 0
        for next_action in self.action_space:
            future_fe += self._calculate_expected_free_energy(
                next_state, next_action, depth + 1
            )
        future_fe /= len(self.action_space)

        total_fe = (
            immediate_fe * 0.05  # Further reduce the impact of immediate free energy
            + info_gain * 2  # Increase the importance of information gain
            - immediate_reward
            + self.discount_factor * future_fe
        )

        return min(total_fe, self.max_fe)

    def learn(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        reward: float,
    ):
        super().learn(state, action, next_state, reward)

        # Additional learning step: update beliefs based on reward
        reward_error = (
            reward - self.root_belief.mean[0]
        )  # Assume first dimension represents value
        self.root_belief.update(np.array([reward_error, 0]), self.learning_rate * 0.1)

    def take_action(self, state: np.ndarray) -> np.ndarray:
        q_values = self._calculate_q_values(state)
        if np.random.random() < self.exploration_rate:
            return self.action_space[np.random.choice(len(self.action_space))]
        else:
            # Softmax action selection
            exp_q_values = np.exp(q_values - np.max(q_values))
            probs = exp_q_values / np.sum(exp_q_values)
            return self.action_space[np.random.choice(len(self.action_space), p=probs)]


# from pydantic import Field


class FurtherImprovedAdvancedFEPAgent(ImprovedAdvancedFEPAgent):
    learning_rate: float = Field(default=0.001)
    discount_factor: float = Field(default=0.99)
    exploration_decay: float = Field(default=0.9995)
    min_exploration_rate: float = Field(default=0.01)
    reward_buffer: list = Field(default_factory=list)
    reward_buffer_size: int = Field(default=100)
    exploration_rate: float = Field(default=0.3)  # Initial exploration rate

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reward_buffer = []

    def learn(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        reward: float,
    ):
        super().learn(state, action, next_state, reward)

        # Store reward in buffer
        self.reward_buffer.append(reward)
        if len(self.reward_buffer) > self.reward_buffer_size:
            self.reward_buffer.pop(0)

        # Normalize reward based on recent history
        mean_reward = np.mean(self.reward_buffer)
        std_reward = (
            np.std(self.reward_buffer) + 1e-5
        )  # Add small epsilon to avoid division by zero
        normalized_reward = (reward - mean_reward) / std_reward

        # Update beliefs based on normalized reward
        reward_error = normalized_reward - self.root_belief.mean[0]
        self.root_belief.update(np.array([reward_error, 0]), self.learning_rate * 0.1)

    def take_action(self, state: np.ndarray) -> np.ndarray:
        self.exploration_rate = max(
            self.min_exploration_rate, self.exploration_rate * self.exploration_decay
        )

        q_values = self._calculate_q_values(state)
        if np.random.random() < self.exploration_rate:
            return self.action_space[np.random.choice(len(self.action_space))]
        else:
            # Softmax action selection with temperature
            temperature = max(
                0.1, self.exploration_rate
            )  # Adjust temperature based on exploration rate
            exp_q_values = np.exp((q_values - np.max(q_values)) / temperature)
            probs = exp_q_values / np.sum(exp_q_values)
            return self.action_space[np.random.choice(len(self.action_space), p=probs)]


class ExperienceReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        return self.network(x)


class DQNFEPAgent(FurtherImprovedAdvancedFEPAgent):
    state_dim: int = Field(...)
    action_dim: int = Field(...)
    action_space: np.ndarray = Field(...)
    replay_buffer: ExperienceReplayBuffer = Field(
        default_factory=lambda: ExperienceReplayBuffer()
    )
    batch_size: int = Field(default=64)
    gamma: float = Field(default=0.99)
    epsilon: float = Field(default=1.0)
    epsilon_decay: float = Field(default=0.995)
    epsilon_min: float = Field(default=0.01)
    learning_rate: float = Field(default=0.001)
    tau: float = Field(default=0.001)  # For soft update of target network
    device: torch.device = Field(
        default_factory=lambda: torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )

    q_network: DQN = Field(default=None)
    target_network: DQN = Field(default=None)
    optimizer: optim.Adam = Field(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data):
        super().__init__(**data)
        self.q_network = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_network = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

    def take_action(self, state: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.epsilon:
            return self.action_space[np.random.choice(len(self.action_space))]
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return self.action_space[q_values.argmax().item()]

    def learn(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        reward: float,
        done: bool,
    ):
        self.replay_buffer.add(state, action, reward, next_state, done)

        if len(self.replay_buffer) < self.batch_size:
            return

        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(
            [
                np.where((self.action_space == action).all(axis=1))[0][0]
                for action in actions
            ]
        ).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update_target_network()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def soft_update_target_network(self):
        for target_param, local_param in zip(
            self.target_network.parameters(), self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )
