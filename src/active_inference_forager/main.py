import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import torch
import os
from active_inference_forager.environments.philosophy_dialogue_environment import (
    PhilosophyDialogueEnvironment,
)
from active_inference_forager.agents.philosophy_tutor_agent import PhilosophyTutorAgent


import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def train_agent(agent, env, n_episodes, log_interval=100):
    rewards_history = []
    action_counts = defaultdict(int)
    episode_lengths = []

    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        episode_actions = []

        while True:
            action = agent.take_action(state)
            next_state, reward, done = env.step(action)

            agent.learn(state, action, next_state, reward, done)

            state = next_state
            episode_reward += reward
            episode_actions.append(action)
            action_counts[action] += 1

            if done:
                break

        rewards_history.append(episode_reward)
        episode_lengths.append(len(episode_actions))

        if episode % log_interval == 0:
            avg_reward = np.mean(rewards_history[-log_interval:])
            avg_length = np.mean(episode_lengths[-log_interval:])
            print(f"Episode {episode}/{n_episodes}")
            print(f"Average Reward: {avg_reward:.2f}")
            print(f"Average Episode Length: {avg_length:.2f}")
            print("Action Distribution:")
            total_actions = sum(action_counts.values())
            for action, count in action_counts.items():
                print(f"  {action}: {count/total_actions:.2%}")
            print("\n")

            # Optionally, plot metrics here
            # plot_metrics(rewards_history, episode_lengths, action_counts)

    return rewards_history, episode_lengths, action_counts


def plot_metrics(rewards, lengths, action_counts):
    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.plot(rewards)
    plt.title("Rewards per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")

    plt.subplot(132)
    plt.plot(lengths)
    plt.title("Episode Lengths")
    plt.xlabel("Episode")
    plt.ylabel("Steps")

    plt.subplot(133)
    actions = list(action_counts.keys())
    counts = list(action_counts.values())
    plt.bar(actions, counts)
    plt.title("Action Distribution")
    plt.xlabel("Action")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.show()


def test_agent(
    agent: PhilosophyTutorAgent,
    env: PhilosophyDialogueEnvironment,
    n_episodes: int = 100,
) -> Tuple[float, float]:
    test_rewards = []
    test_steps = []
    original_exploration_rate = agent.exploration_rate
    agent.exploration_rate = 0  # Disable exploration during testing

    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False

        while not done:
            action = agent.take_action(state)
            next_state, reward, done = env.step(action)
            state = next_state
            episode_reward += reward
            episode_steps += 1

        test_rewards.append(episode_reward)
        test_steps.append(episode_steps)

        print(
            f"Test Episode {episode}: Reward = {episode_reward:.2f}, Steps = {episode_steps}"
        )

    agent.exploration_rate = (
        original_exploration_rate  # Restore the original exploration rate
    )

    avg_reward = np.mean(test_rewards)
    avg_steps = np.mean(test_steps)
    consistency = np.sum([r > 0.5 for r in test_rewards]) / n_episodes

    print(
        f"Test Results - Avg Reward: {avg_reward:.2f}, Avg Steps: {avg_steps:.2f}, Consistency: {consistency:.2f}"
    )
    return avg_reward, avg_steps


def simulate_conversation(
    agent: PhilosophyTutorAgent, env: PhilosophyDialogueEnvironment, max_turns: int = 10
):
    state = env.reset()
    done = False
    turn = 0

    print("Starting conversation simulation:")
    print("----------------------------------")

    while not done and turn < max_turns:
        action = agent.take_action(state)
        response = agent.generate_response(action, state)
        print(f"Agent: {response}")

        user_input = input("User: ")

        next_state, reward, done = env.step(action)

        agent.update_belief(user_input)
        state = next_state
        turn += 1

        print(f"Turn {turn}: Reward = {reward:.2f}")
        print(f"Current topic: {env.current_topic}")
        print(f"User engagement: {env.user_engagement:.2f}")
        print("----------------------------------")

    print("Conversation ended.")
    print(f"Final user understanding: {env.user_understanding}")
    print(f"Final user interests: {env.user_interests}")


def save_agent(agent: PhilosophyTutorAgent, path: str):
    torch.save(
        {
            "q_network_state_dict": agent.q_network.state_dict(),
            "target_network_state_dict": agent.target_network.state_dict(),
            "optimizer_state_dict": agent.optimizer.state_dict(),
            "exploration_rate": agent.exploration_rate,
            "total_steps": agent.total_steps,
            "knowledge_base": agent.knowledge_base,
        },
        path,
    )
    print(f"Agent saved to {path}")


def load_agent(agent: PhilosophyTutorAgent, path: str):
    checkpoint = torch.load(path)
    agent.q_network.load_state_dict(checkpoint["q_network_state_dict"])
    agent.target_network.load_state_dict(checkpoint["target_network_state_dict"])
    agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    agent.exploration_rate = checkpoint["exploration_rate"]
    agent.total_steps = checkpoint["total_steps"]
    agent.knowledge_base = checkpoint["knowledge_base"]
    print(f"Agent loaded from {path}")


def main():
    env = PhilosophyDialogueEnvironment()
    initial_state = env.reset()
    state_dim = initial_state.shape[0]
    action_dim = len(env.action_space)

    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")

    agent = PhilosophyTutorAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=1e-3,
        discount_factor=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        batch_size=64,
        max_kl=10.0,
        max_fe=100.0,
        belief_regularization=0.01,
    )

    print(f"Agent root belief mean shape: {agent.root_belief.mean.shape}")
    print(f"Agent root belief precision shape: {agent.root_belief.precision.shape}")

    model_path = "./models/philosophy_tutor_agent.pth"

    print("Starting training...")
    n_episodes = 1000  # Reduced number of episodes for faster training
    rewards_history, steps_history, action_counts = train_agent(agent, env, n_episodes)
    plot_metrics(rewards_history, steps_history, action_counts)

    save_agent(agent, model_path)

    # Test the agent
    test_agent(agent, env)

    # Simulate a conversation with the trained agent
    simulate_conversation(agent, env)


if __name__ == "__main__":
    main()
