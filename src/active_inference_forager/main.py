import numpy as np
import logging
import matplotlib.pyplot as plt
from typing import Tuple, List
from active_inference_forager.agents.environment import SimpleEnvironment
from active_inference_forager.agents.advanced_fep_agent import (
    DQNFEPAgent,
)


def train_agent(
    agent: DQNFEPAgent,
    env: SimpleEnvironment,
    n_episodes: int,
    early_stop_threshold: float = 1e-5,
    patience: int = 1000,
) -> Tuple[List[float], List[int]]:
    rewards_history = []
    steps_history = []
    best_reward = float("-inf")
    patience_counter = 0

    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done:
            action = agent.take_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, next_state, reward, done)
            state = next_state
            total_reward += reward
            steps += 1

        rewards_history.append(total_reward)
        steps_history.append(steps)

        # Early stopping
        if total_reward > best_reward:
            best_reward = total_reward
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience and best_reward > early_stop_threshold:
            print(f"Early stopping at episode {episode}")
            break

        if episode % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            avg_steps = np.mean(steps_history[-10:])
            print(
                f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Avg Steps: {avg_steps:.1f}, Epsilon: {agent.epsilon:.4f}"
            )

    return rewards_history, steps_history


def main():
    env = SimpleEnvironment()
    agent = DQNFEPAgent(
        state_dim=env.state_dim,
        action_dim=len(env.action_space),
        action_space=env.action_space,
        learning_rate=1e-2,
        discount_factor=0.99,
        exploration_decay=0.999,
    )

    n_episodes = 250000  # Increased number of episodes for DQN learning
    rewards_history, steps_history = train_agent(agent, env, n_episodes)

    # Plot results
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(rewards_history)
    plt.title("Rewards per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")

    plt.subplot(1, 3, 2)
    plt.plot(steps_history)
    plt.title("Steps per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Steps")

    plt.subplot(1, 3, 3)
    plt.plot(np.convolve(rewards_history, np.ones(100) / 100, mode="valid"))
    plt.title("Moving Average Reward (100 episodes)")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
