import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Any, Tuple
import random


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

    def update(self, state: Dict[str, Any], action: str, response: str):
        if "healthy" in action.lower():
            self.preferences["health_conscious"] = (
                self.preferences.get("health_conscious", 0) + 0.1
            )
        self.trust_level = min(1.0, self.trust_level + 0.01)


class EnhancedChatbotEnv:
    def __init__(
        self, goals: List[ConfigurableGoal], constraints: List[EthicalConstraint]
    ):
        self.goals = goals
        self.constraints = constraints
        self.user_model = UserModel()
        self.state = self.reset()

    def reset(self):
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
        done = self.state["current_goal"].completion_condition(self.state)
        return self.state, reward, done, {}

    def simulate_user_response(self, action: str) -> str:
        responses = [
            "That's interesting. Tell me more.",
            "I'm not sure I agree with that.",
            "How does that relate to my health?",
            "Can you explain that in simpler terms?",
        ]
        return random.choice(responses)

    def calculate_reward(self, action: str, user_response: str) -> float:
        reward = self.state["current_goal"].reward_function(
            self.state, action, user_response
        )
        for constraint in self.constraints:
            if constraint.check_function(self.state, action, user_response):
                reward -= constraint.penalty
        return reward


class LanguageModel(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, text: str):
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)


class PolicyNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.action_head = nn.Linear(128, action_size)
        self.value_head = nn.Linear(128, 1)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = nn.functional.leaky_relu(self.fc1(x))
        x = nn.functional.leaky_relu(self.fc2(x))
        action_logits = self.action_head(x)
        log_probs = self.log_softmax(action_logits)
        action_probs = torch.exp(log_probs)
        state_value = self.value_head(x)
        return action_probs, log_probs, state_value.squeeze(-1)


class EnhancedPPOAgent:
    def __init__(
        self, state_size: int, action_size: int, language_model: LanguageModel
    ):
        self.policy = PolicyNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)
        self.language_model = language_model
        self.action_space = self.generate_action_space(action_size)

        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def generate_action_space(self, action_size: int) -> List[str]:
        return [f"Action {i}" for i in range(action_size)]

    def get_action(
        self, state: Dict[str, Any]
    ) -> Tuple[str, torch.Tensor, torch.Tensor]:
        state_vector = self.state_to_vector(state)
        action_probs, log_probs, state_value = self.policy(state_vector)
        
        print("Debug - action_probs:", action_probs)
        print("Debug - log_probs:", log_probs)
        print("Debug - state_value:", state_value)
        
        if torch.isnan(action_probs).any() or torch.isinf(action_probs).any():
            print("Warning: NaN or Inf detected in action_probs")
            action_probs = torch.ones_like(action_probs) / len(action_probs)
            log_probs = torch.log(action_probs)

        if random.random() < 0.1:  # 10% chance of random action
            action_index = random.randint(0, len(self.action_space) - 1)
        else:
            action_index = torch.multinomial(action_probs, 1).item()

        return self.action_space[action_index], log_probs[action_index], state_value

    def state_to_vector(self, state: Dict[str, Any]) -> torch.Tensor:
        last_turn = (
            state["conversation_history"][-1]
            if state["conversation_history"]
            else ("", "")
        )
        text = " ".join(last_turn)
        vector = self.language_model(text).detach().squeeze(0)
        print("Debug - state vector:", vector)
        return vector

    def store_transition(self, state, action, reward, log_prob, value, done):
        self.states.append(state)
        self.actions.append(self.action_space.index(action))
        self.rewards.append(reward)
        self.values.append(value.detach())
        self.log_probs.append(log_prob.detach())
        self.dones.append(done)

    def train(self):
        if not self.values:
            return

        returns = self.compute_returns(self.values[-1])
        returns = torch.tensor(returns)

        with torch.no_grad():
            advantages = returns - torch.cat([v.view(-1) for v in self.values])
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(5):
            for state, action_idx, old_log_prob, advantage, return_ in zip(
                self.states, self.actions, self.log_probs, advantages, returns
            ):
                state_vector = self.state_to_vector(state)
                _, log_probs, value = self.policy(state_vector)
                new_log_prob = log_probs[action_idx]

                ratio = (new_log_prob - old_log_prob).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 0.8, 1.2) * advantage
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = 0.5 * (return_ - value).pow(2).mean()

                loss = actor_loss + 0.5 * critic_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()

                print("Debug - loss:", loss.item())
                print("Debug - actor_loss:", actor_loss.item())
                print("Debug - critic_loss:", critic_loss.item())

        self.scheduler.step()
        self.states, self.actions, self.rewards = [], [], []
        self.values, self.log_probs, self.dones = [], [], []

    def compute_returns(self, last_value, gamma=0.99):
        returns = []
        R = last_value
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            R = reward + gamma * R * (1 - done)
            returns.insert(0, R)
        return returns


def main():
    goals = [
        ConfigurableGoal(
            "promote_healthy_eating",
            lambda state, action, response: 1 if "healthy" in action.lower() else 0,
            lambda state: len(state["conversation_history"]) >= 10,
        ),
        ConfigurableGoal(
            "build_trust",
            lambda state, action, response: (
                0.5 if state["user_model"].trust_level > 0.7 else 0
            ),
            lambda state: state["user_model"].trust_level >= 0.9,
        ),
    ]

    constraints = [
        EthicalConstraint(
            "avoid_deception",
            lambda state, action, response: "lie" in action.lower()
            or "deceive" in action.lower(),
            penalty=2.0,
        ),
        EthicalConstraint(
            "respect_privacy",
            lambda state, action, response: "personal" in action.lower()
            and state["user_model"].trust_level < 0.8,
            penalty=1.5,
        ),
    ]

    env = EnhancedChatbotEnv(goals, constraints)
    language_model = LanguageModel("distilbert-base-uncased")
    agent = EnhancedPPOAgent(state_size=768, action_size=100, language_model=language_model)

    num_episodes = 5
    max_steps_per_episode = 50
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        step = 0

        while not done and step < max_steps_per_episode:
            action, log_prob, state_value = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, log_prob, state_value, done)
            state = next_state
            episode_reward += reward
            step += 1

        agent.train()
        print(
            f"Episode {episode}/{num_episodes}, Reward: {episode_reward:.2f}, Steps: {step}"
        )

        if episode > 10 and episode_reward > 5:
            print("Early stopping condition met. Training complete.")
            break

    print("Training Complete")


if __name__ == "__main__":
    main()
