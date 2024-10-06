import numpy as np
from active_inference_forager.agents.environment import Environment
from active_inference_forager.agents.advanced_fep_agent import AdvancedFEPAgent
from active_inference_forager.agents.belief_node import BeliefNode
from pydantic import Field


class SimpleEnvironment(Environment):
    max_steps: int = Field(default=100)
    steps: int = Field(default=0)

    def __init__(self, **data):
        super().__init__(state_dim=2, action_dim=2, **data)
        self.state = np.zeros(2)
        self.reset()

    def step(self, action):
        self.state += action
        distance = np.linalg.norm(self.state)
        reward = -distance * 0.1  # Scaled reward
        self.steps += 1
        done = distance < 0.1 or self.steps >= self.max_steps
        return self.state, reward, done

    def reset(self):
        self.state = np.random.uniform(-1, 1, 2)
        self.steps = 0
        return self.state


def main():
    env = SimpleEnvironment()
    action_space = np.array(
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
    agent = AdvancedFEPAgent(
        root_belief=BeliefNode(mean=np.zeros(2), precision=np.eye(2)),
        action_space=action_space,
        state_dim=2,
        learning_rate=0.1,  # Increased from 0.01
        exploration_factor=0.2,  # Increased from 0.1
        discount_factor=0.99,  # Slightly increased
    )

    n_episodes = 1000
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        step = 0
        while not done:
            action = agent.take_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, next_state, reward)
            state = next_state
            total_reward += reward
            step += 1
            if step % 10 == 0:
                print(
                    f"Episode {episode + 1}/{n_episodes}, Step {step}, Total Reward: {total_reward:.2f}"
                )

        print(
            f"Episode {episode + 1}/{n_episodes} finished. Total Reward: {total_reward:.2f}"
        )

    print("Training completed.")


if __name__ == "__main__":
    main()
