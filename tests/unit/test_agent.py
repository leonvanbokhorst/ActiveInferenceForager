import numpy as np
import torch
from active_inference_forager.agents.advanced_fep_agent import DQNFEPAgent


def test_dqnfep_agent_initialization():
    agent = DQNFEPAgent(state_dim=2, action_dim=4, action_space=np.array([[0, 1], [0, -1], [1, 0], [-1, 0]]))
    assert agent is not None
    assert agent.state_dim == 2
    assert agent.action_dim == 4
    assert agent.q_network is not None
    assert agent.target_network is not None
    assert agent.optimizer is not None


def test_dqnfep_agent_belief_update():
    agent = DQNFEPAgent(state_dim=2, action_dim=4, action_space=np.array([[0, 1], [0, -1], [1, 0], [-1, 0]]))
    initial_belief = agent.root_belief.mean.copy()
    observation = np.array([1.0, 2.0])
    agent.update_belief(observation)
    updated_belief = agent.root_belief.mean
    assert not np.array_equal(updated_belief, initial_belief)


def test_dqnfep_agent_take_action():
    agent = DQNFEPAgent(state_dim=2, action_dim=4, action_space=np.array([[0, 1], [0, -1], [1, 0], [-1, 0]]))
    state = np.array([0.5, 0.5])
    action = agent.take_action(state)
    assert action in agent.action_space


def test_dqnfep_agent_learn():
    agent = DQNFEPAgent(state_dim=2, action_dim=4, action_space=np.array([[0, 1], [0, -1], [1, 0], [-1, 0]]), batch_size=4)
    state = np.array([0.5, 0.5])
    action = np.array([0, 1])
    next_state = np.array([0.5, 1.5])
    reward = 1.0
    done = False

    # Fill the replay buffer
    for _ in range(10):
        agent.replay_buffer.add(state, action, reward, next_state, done)

    initial_q_values = agent.q_network(torch.FloatTensor(state).unsqueeze(0)).detach().numpy()

    # Perform multiple learning steps
    for _ in range(10):
        agent.learn(state, action, next_state, reward, done)

    updated_q_values = agent.q_network(torch.FloatTensor(state).unsqueeze(0)).detach().numpy()

    # Check if Q-values have changed with a small tolerance
    assert not np.allclose(initial_q_values, updated_q_values, atol=1e-4)


def test_dqnfep_agent_exploration_decay():
    agent = DQNFEPAgent(state_dim=2, action_dim=4, action_space=np.array([[0, 1], [0, -1], [1, 0], [-1, 0]]))
    initial_exploration_rate = agent.exploration_rate
    agent.decay_exploration()
    assert agent.exploration_rate < initial_exploration_rate


def test_dqnfep_agent_free_energy_update():
    agent = DQNFEPAgent(state_dim=2, action_dim=4, action_space=np.array([[0, 1], [0, -1], [1, 0], [-1, 0]]))
    initial_free_energy = agent.free_energy
    observation = np.array([1.0, 2.0])
    agent.update_belief(observation)
    agent.update_free_energy()
    assert agent.free_energy != initial_free_energy
