import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import torch
import os
from active_inference_forager.environments.chat_environment import ChatEnvironment
from active_inference_forager.agents.dqn_fep_agent import DQNFEPAgent


def train_agent(
    agent: DQNFEPAgent,
    env: ChatEnvironment,
    n_episodes: int,
    early_stop_threshold: float = 0.8,
    patience: int = 1000,
) -> Tuple[List[float], List[int]]:
    rewards_history = []
    steps_history = []
    best_reward = float("-inf")
    patience_counter = 0
    best_consistency = 0
    consistency_threshold = 0.9  # 90% of episodes should be good

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

        # Early stopping based on both performance and consistency
        if len(rewards_history) >= 100:
            recent_rewards = rewards_history[-100:]
            avg_reward = np.mean(recent_rewards)
            consistency = np.sum([r > 0.5 for r in recent_rewards]) / 100

            if avg_reward > best_reward and consistency > consistency_threshold:
                best_reward = avg_reward
                best_consistency = consistency
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
                f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Avg Steps: {avg_steps:.1f}, Exploration Rate: {agent.exploration_rate:.4f}"
            )

    return rewards_history, steps_history


def test_agent(agent: DQNFEPAgent, env: ChatEnvironment, n_episodes: int = 100) -> Tuple[float, float]:
    test_rewards = []
    test_steps = []
    agent.q_network.eval()  # Set the network to evaluation mode
    original_exploration_rate = agent.exploration_rate
    agent.exploration_rate = 0  # Disable exploration during testing

    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False

        while not done:
            with torch.no_grad():
                action = agent.take_action(state)
            next_state, reward, done = env.step(action)
            state = next_state
            episode_reward += reward
            episode_steps += 1

        test_rewards.append(episode_reward)
        test_steps.append(episode_steps)

        print(f"Test Episode {episode}: Reward = {episode_reward:.2f}, Steps = {episode_steps}")

    agent.exploration_rate = original_exploration_rate  # Restore the original exploration rate

    avg_reward = np.mean(test_rewards)
    avg_steps = np.mean(test_steps)
    consistency = np.sum([r > 0.5 for r in test_rewards]) / n_episodes

    print(f"Test Results - Avg Reward: {avg_reward:.2f}, Avg Steps: {avg_steps:.2f}, Consistency: {consistency:.2f}")
    return avg_reward, avg_steps


def simulate_conversation(agent: DQNFEPAgent, env: ChatEnvironment, max_turns: int = 10):
    state = env.reset()
    done = False
    turn = 0

    print("Starting conversation simulation:")
    print("----------------------------------")

    while not done and turn < max_turns:
        action = agent.take_action(state)
        print(f"Agent: {agent.interpret_action(action)}")
        
        # Simulate user input (in a real scenario, this would come from actual user input)
        user_input = input("User: ")
        
        # Process user input and update state
        user_features = agent.process_user_input(user_input)
        next_state, reward, done = env.step(action)
        
        # Combine environment state with user input features
        next_state = np.concatenate([next_state, user_features])
        
        agent.learn(state, action, next_state, reward, done)
        state = next_state
        turn += 1

        print(f"Turn {turn}: Reward = {reward:.2f}")
        print("----------------------------------")

    print("Conversation ended.")
    print(f"Final user satisfaction: {env.user_satisfaction:.2f}")
    print(f"Final task completion: {env.task_completion:.2f}")


def save_agent(agent: DQNFEPAgent, path: str):
    torch.save({
        'q_network_state_dict': agent.q_network.state_dict(),
        'target_network_state_dict': agent.target_network.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'exploration_rate': agent.exploration_rate,
        'total_steps': agent.total_steps,
    }, path)
    print(f"Agent saved to {path}")


def load_agent(agent: DQNFEPAgent, path: str):
    checkpoint = torch.load(path)
    agent.q_network.load_state_dict(checkpoint['q_network_state_dict'])
    agent.target_network.load_state_dict(checkpoint['target_network_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    agent.exploration_rate = checkpoint['exploration_rate']
    agent.total_steps = checkpoint['total_steps']
    print(f"Agent loaded from {path}")


def main():
    env = ChatEnvironment()
    agent = DQNFEPAgent(
        state_dim=env.state_dim,
        action_dim=len(env.action_space),
        action_space=env.action_space,
        learning_rate=1e-3,
        discount_factor=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        batch_size=64,
        reward_buffer_size=100,
        max_kl=10.0,
        max_fe=100.0,
        belief_regularization=0.01,
    )

    model_path = "dqn_fep_agent.pth"

    # Force retraining
    print("Starting training...")
    n_episodes = 10000  # Reduced number of episodes for faster training
    rewards_history, steps_history = train_agent(agent, env, n_episodes)

    # Save the trained agent
    save_agent(agent, model_path)

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

    # Test the agent
    test_agent(agent, env)

    # Simulate a conversation with the trained agent
    simulate_conversation(agent, env)


if __name__ == "__main__":
    main()
