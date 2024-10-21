import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Any, Tuple
import random
import numpy as np


class ConfigurableGoal:
    def __init__(
        self, name: str, reward_function: callable, completion_condition: callable
    ):
        self.name = name
        self.reward_function = reward_function
        self.completion_condition = completion_condition


class EthicalConstraint:
    def __init__(self, name: str, check_function: callable, penalty: float):
        self.name = name
        self.check_function = check_function
        self.penalty = penalty


class UserModel:
    def __init__(self):
        self.knowledge = {}
        self.preferences = {}
        self.trust_level = 0.5
        self.engagement_level = 0.5
        self.mood = 0.5

    def update(self, state: Dict[str, Any], action: str, response: str):
        if "healthy" in action.lower():
            self.preferences["health_conscious"] = min(
                1.0, self.preferences.get("health_conscious", 0) + 0.1
            )
        self.trust_level = min(1.0, self.trust_level + 0.01)
        self.engagement_level = min(
            1.0, self.engagement_level + 0.05 * (1 if len(response) > 20 else -1)
        )
        self.mood = min(
            1.0,
            max(
                0.0, self.mood + 0.1 * (1 if "interesting" in response.lower() else -1)
            ),
        )


class EnhancedChatbotEnv:
    def __init__(
        self, goals: List[ConfigurableGoal], constraints: List[EthicalConstraint]
    ):
        self.goals = goals
        self.constraints = constraints
        self.user_model = UserModel()
        self.state = self.reset()
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        return {
            "conversation_history": [],
            "current_goal": random.choice(self.goals),
            "user_model": self.user_model,
        }

    def step(self, action: str):
        user_response = self.simulate_user_response(action)
        self.state["conversation_history"].append((action, user_response))
        self.user_model.update(self.state, action, user_response)
        reward = self.calculate_reward(action, user_response)
        self.step_count += 1
        done = (
            self.state["current_goal"].completion_condition(self.state)
            or self.step_count >= 20
        )
        return self.state, reward, done, {}

    def simulate_user_response(self, action: str) -> str:
        responses = [
            "That's interesting. Tell me more.",
            "I'm not sure I agree with that.",
            "How does that relate to my health?",
            "Can you explain that in simpler terms?",
            "I never thought about it that way before.",
            "That's helpful information, thanks!",
            "I'm feeling a bit overwhelmed with all this health talk.",
            "Do you have any practical tips I can try?",
        ]
        return random.choice(responses)

    def calculate_reward(self, action: str, user_response: str) -> float:
        base_reward = self.state["current_goal"].reward_function(
            self.state, action, user_response
        )
        engagement_bonus = 0.1 * self.user_model.engagement_level
        trust_bonus = 0.1 * self.user_model.trust_level
        mood_bonus = 0.1 * self.user_model.mood

        reward = base_reward + engagement_bonus + trust_bonus + mood_bonus

        for constraint in self.constraints:
            if constraint.check_function(self.state, action, user_response):
                reward -= constraint.penalty
        return reward


class PolicyNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.action_head = nn.Linear(128, action_size)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        action_probs = torch.softmax(self.action_head(x), dim=-1)
        state_value = self.value_head(x)
        return action_probs, state_value.squeeze(-1)


class EnhancedPPOAgent:
    def __init__(
        self, state_size: int, action_size: int, goals: List[ConfigurableGoal]
    ):
        self.policy = PolicyNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        self.action_space = self.generate_action_space()
        self.epsilon = 0.1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.goals = goals

    def generate_action_space(self) -> List[str]:
        return [
            "Tell me about healthy eating habits.",
            "What's your favorite healthy food?",
            "How often do you exercise?",
            "Do you have any health concerns?",
            "Let's discuss the benefits of a balanced diet.",
            "How do you manage stress in your daily life?",
            "What are your thoughts on mindfulness and meditation?",
            "Have you considered trying a new physical activity recently?",
        ]

    def get_action(
        self, state: Dict[str, Any]
    ) -> Tuple[str, torch.Tensor, torch.Tensor]:
        state_vector = self.state_to_vector(state)
        action_probs, state_value = self.policy(state_vector)

        if random.random() < self.epsilon:
            action_index = random.randint(0, len(self.action_space) - 1)
        else:
            action_index = torch.multinomial(action_probs, 1).item()

        return self.action_space[action_index], action_probs[action_index], state_value

    def state_to_vector(self, state: Dict[str, Any]) -> torch.Tensor:
        features = [
            len(state["conversation_history"]),
            state["user_model"].trust_level,
            state["user_model"].preferences.get("health_conscious", 0),
            state["user_model"].engagement_level,
            state["user_model"].mood,
        ]

        # Add one-hot encoding of the current goal
        goal_encoding = [0] * len(self.goals)
        goal_index = next(
            i
            for i, goal in enumerate(self.goals)
            if goal.name == state["current_goal"].name
        )
        goal_encoding[goal_index] = 1
        features.extend(goal_encoding)

        # Add features from the last conversation turn
        if state["conversation_history"]:
            last_action, last_response = state["conversation_history"][-1]
            features.append(len(last_response))
            features.append(1 if "interesting" in last_response.lower() else 0)
        else:
            features.extend([0, 0])  # Padding for no conversation history

        return torch.tensor(features, dtype=torch.float32)

    def update(self, states, actions, rewards, dones):
        returns = self.compute_returns(rewards, dones)
        states = torch.stack([self.state_to_vector(s) for s in states])
        action_probs, state_values = self.policy(states)

        advantages = returns - state_values.detach()

        action_indices = torch.tensor([self.action_space.index(a) for a in actions])
        old_probs = (
            action_probs.gather(1, action_indices.unsqueeze(1)).squeeze(1).detach()
        )

        for _ in range(5):  # PPO epochs
            new_action_probs, new_state_values = self.policy(states)
            new_probs = new_action_probs.gather(1, action_indices.unsqueeze(1)).squeeze(
                1
            )

            ratio = new_probs / (old_probs + 1e-8)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(new_state_values, returns)

            loss = actor_loss + 0.5 * critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Learning rate decay
        for param_group in self.optimizer.param_groups:
            param_group["lr"] *= 0.995

    def compute_returns(self, rewards, dones, gamma=0.99):
        returns = []
        R = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            R = reward + gamma * R * (1 - done)
            returns.insert(0, R)
        return torch.tensor(returns)


def main():
    goals = [
        ConfigurableGoal(
            "promote_healthy_eating",
            lambda state, action, response: 0.2 if "healthy" in action.lower() else 0,
            lambda state: state["user_model"].preferences.get("health_conscious", 0)
            >= 0.7,
        ),
        ConfigurableGoal(
            "build_trust",
            lambda state, action, response: 0.1 * state["user_model"].trust_level,
            lambda state: state["user_model"].trust_level >= 0.9,
        ),
        ConfigurableGoal(
            "improve_mood",
            lambda state, action, response: (
                0.2 if "interesting" in response.lower() else -0.1
            ),
            lambda state: state["user_model"].mood >= 0.8,
        ),
    ]

    constraints = [
        EthicalConstraint(
            "avoid_deception",
            lambda state, action, response: "lie" in action.lower()
            or "deceive" in action.lower(),
            penalty=0.5,
        ),
        EthicalConstraint(
            "respect_privacy",
            lambda state, action, response: "personal" in action.lower()
            and state["user_model"].trust_level < 0.8,
            penalty=0.3,
        ),
    ]

    env = EnhancedChatbotEnv(goals, constraints)
    initial_state = env.reset()
    dummy_agent = EnhancedPPOAgent(1, 1, goals)
    state_size = len(dummy_agent.state_to_vector(initial_state))
    action_size = len(dummy_agent.action_space)
    agent = EnhancedPPOAgent(
        state_size=state_size, action_size=action_size, goals=goals
    )

    num_episodes = 100
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False

        states, actions, rewards, dones = [], [], [], []

        while not done:
            action, _, _ = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

            state = next_state
            episode_reward += reward

        agent.update(states, actions, rewards, dones)
        print(
            f"Episode {episode}/{num_episodes}, Reward: {episode_reward:.2f}, Steps: {env.step_count}, Epsilon: {agent.epsilon:.3f}"
        )

        if episode > 500 and episode_reward > 3.0:
            print("Early stopping condition met. Training complete.")
            break

    print("Training Complete")


if __name__ == "__main__":
    main()
