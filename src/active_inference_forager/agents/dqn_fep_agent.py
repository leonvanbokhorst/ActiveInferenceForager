import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple
from pydantic import Field, ConfigDict
from collections import deque
import random

from active_inference_forager.agents.base_agent import BaseAgent
from active_inference_forager.agents.belief_node import BeliefNode
from active_inference_forager.utils.numpy_fields import NumpyArrayField

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

class DQNFEPAgent(BaseAgent):
    # FEP-related parameters
    max_kl: float = Field(default=10.0)
    max_fe: float = Field(default=100.0)
    epsilon: float = Field(default=1e-7)
    min_precision: float = Field(default=0.01)
    max_precision: float = Field(default=100.0)
    max_mean: float = Field(default=1e2)
    max_depth: int = Field(default=2)
    belief_regularization: float = Field(default=0.01)
    max_belief_value: float = Field(default=100.0)
    free_energy: float = Field(default=0.0)

    # DQN-related parameters
    replay_buffer: ExperienceReplayBuffer = Field(default_factory=lambda: ExperienceReplayBuffer())
    batch_size: int = Field(default=64)
    gamma: float = Field(default=0.99)
    epsilon_start: float = Field(default=1.0)
    epsilon_end: float = Field(default=0.01)
    epsilon_decay: float = Field(default=0.995)
    tau: float = Field(default=0.001)  # For soft update of target network
    device: torch.device = Field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    q_network: DQN = Field(default=None)
    target_network: DQN = Field(default=None)
    optimizer: optim.Adam = Field(default=None)

    # Additional parameters
    reward_buffer: List[float] = Field(default_factory=list)
    reward_buffer_size: int = Field(default=100)
    episode_rewards: List[float] = Field(default_factory=list)
    episode_lengths: List[int] = Field(default_factory=list)
    total_steps: int = Field(default=0)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data):
        super().__init__(**data)
        self.initialize_belief_and_action_space()
        self.q_network = DQN(self.state_dim, len(self.action_space)).to(self.device)
        self.target_network = DQN(self.state_dim, len(self.action_space)).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.exploration_rate = self.epsilon_start

    def take_action(self, state: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.exploration_rate:
            return self.action_space[np.random.choice(len(self.action_space))]
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return self.action_space[q_values.argmax().item()]

    def learn(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray, reward: float, done: bool):
        self.replay_buffer.add(state, action, reward, next_state, done)
        self.update_belief(next_state)
        self.update_free_energy()
        self.update_reward_buffer(reward)

        if len(self.replay_buffer) < self.batch_size:
            return

        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor([np.where((self.action_space == action).all(axis=1))[0][0] for action in actions]).to(self.device)
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
        self.decay_exploration()
        self.total_steps += 1

    def update_belief(self, observation: np.ndarray) -> None:
        self._update_belief_recursive(self.root_belief, observation)
        self._regularize_beliefs()

    def _update_belief_recursive(self, node: BeliefNode, observation: np.ndarray):
        prediction_error = observation - node.mean
        node.precision += np.outer(prediction_error, prediction_error) * self.learning_rate
        node.precision = np.clip(node.precision, self.min_precision, self.max_precision)

        precision_inv = np.linalg.inv(node.precision + np.eye(node.dim) * self.epsilon)
        node.mean += precision_inv.dot(prediction_error) * self.learning_rate
        node.mean = np.clip(node.mean, -self.max_belief_value, self.max_belief_value)

        for child in node.children.values():
            self._update_belief_recursive(child, observation)

    def _regularize_beliefs(self):
        self.root_belief.mean *= 1 - self.belief_regularization
        self.root_belief.precision += np.eye(self.root_belief.dim) * self.belief_regularization

    def update_free_energy(self):
        self.free_energy = self._calculate_free_energy_recursive(self.root_belief)

    def _calculate_free_energy_recursive(self, node: BeliefNode) -> float:
        kl_divergence = self._kl_divergence(node)
        for child in node.children.values():
            kl_divergence += self._calculate_free_energy_recursive(child)
        return kl_divergence

    def _kl_divergence(self, node: BeliefNode) -> float:
        prior_mean = np.zeros_like(node.mean)
        prior_precision = np.eye(node.dim)

        try:
            node_cov = np.linalg.inv(node.precision + np.eye(node.dim) * self.epsilon)
            prior_cov = np.linalg.inv(prior_precision + np.eye(node.dim) * self.epsilon)

            kl = 0.5 * (
                np.trace(prior_precision @ node_cov)
                + (prior_mean - node.mean).T @ prior_precision @ (prior_mean - node.mean)
                - node.dim
                + np.log(np.linalg.det(node.precision) / np.linalg.det(prior_precision))
            )
        except np.linalg.LinAlgError:
            return self.max_kl

        return float(min(max(kl, 0), self.max_kl))

    def update_reward_buffer(self, reward: float):
        self.reward_buffer.append(reward)
        if len(self.reward_buffer) > self.reward_buffer_size:
            self.reward_buffer.pop(0)

    def soft_update_target_network(self):
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def decay_exploration(self):
        self.exploration_rate = max(self.epsilon_end, self.exploration_rate * self.epsilon_decay)

    def reset(self):
        self.root_belief = BeliefNode(
            mean=np.zeros(self.state_dim),
            precision=np.eye(self.state_dim) * 0.1,
            max_precision=self.max_precision,
            max_mean=self.max_mean,
        )
        self._build_belief_hierarchy(self.root_belief, 0)
        self.free_energy = 0.0
        self.exploration_rate = self.epsilon_start

        if self.episode_rewards:
            self.episode_rewards.append(0)
            self.episode_lengths.append(0)

    def _build_belief_hierarchy(self, node: BeliefNode, level: int):
        if level >= self.max_depth:
            return
        for i in range(len(self.action_space)):
            child = BeliefNode(
                mean=np.zeros(self.state_dim),
                precision=np.eye(self.state_dim) * (0.1 / (level + 1)),
                level=level + 1,
                max_precision=self.max_precision,
                max_mean=self.max_mean,
            )
            node.children[f"action_{i}"] = child
            self._build_belief_hierarchy(child, level + 1)
