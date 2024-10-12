import pytest
import numpy as np
import torch
from active_inference_forager.agents.dqn_fep_agent import DQNFEPAgent

@pytest.fixture
def agent():
    action_space = ['ask_question', 'provide_information', 'clarify', 'suggest_action', 'express_empathy', 'end_conversation']
    return DQNFEPAgent(state_dim=5, action_dim=len(action_space), action_space=action_space)

def test_agent_initialization(agent):
    assert isinstance(agent, DQNFEPAgent)
    assert agent.state_dim == 5
    assert np.array_equal(agent.action_space, ['ask_question', 'provide_information', 'clarify', 'suggest_action', 'express_empathy', 'end_conversation'])
    assert isinstance(agent.q_network, torch.nn.Module)
    assert isinstance(agent.target_network, torch.nn.Module)
    assert isinstance(agent.optimizer, torch.optim.Adam)
    assert agent.exploration_rate == agent.epsilon_start

def test_take_action(agent):
    state = np.random.rand(5)
    action = agent.take_action(state)
    assert action in agent.action_space

def test_learn(agent):
    state = np.random.rand(5)
    action = 'ask_question'
    next_state = np.random.rand(5)
    reward = 1.0
    done = False

    initial_total_steps = agent.total_steps
    
    agent.learn(state, action, next_state, reward, done)
    
    assert agent.total_steps > initial_total_steps

def test_update_belief(agent):
    observation = np.random.rand(5)
    initial_mean = agent.root_belief.mean.copy()
    initial_precision = agent.root_belief.precision.copy()
    
    agent.update_belief(observation)
    
    assert not np.array_equal(initial_mean, agent.root_belief.mean)
    assert not np.array_equal(initial_precision, agent.root_belief.precision)

def test_update_free_energy(agent):
    initial_free_energy = agent.free_energy
    agent.update_free_energy()
    assert agent.free_energy != initial_free_energy

def test_update_reward_buffer(agent):
    initial_buffer_length = len(agent.reward_buffer)
    agent.update_reward_buffer(1.0)
    assert len(agent.reward_buffer) == initial_buffer_length + 1

def test_decay_exploration(agent):
    initial_exploration_rate = agent.exploration_rate
    agent.decay_exploration()
    assert agent.exploration_rate < initial_exploration_rate

def test_reset(agent):
    agent.free_energy = 10.0
    agent.exploration_rate = 0.1
    agent.reset()
    assert agent.free_energy == 0.0
    assert agent.exploration_rate == agent.epsilon_start

def test_interpret_action(agent):
    action = 'ask_question'
    interpretation = agent.interpret_action(action)
    assert isinstance(interpretation, str)
    assert "ask a question" in interpretation.lower()

def test_process_user_input(agent):
    user_input = "Hello, how are you? I'm feeling great today!"
    features = agent.process_user_input(user_input)
    assert isinstance(features, np.ndarray)
    assert features.shape == (5,)
    assert features[0] == 8  # Number of words
    assert features[1] == 1/8  # Question mark ratio
    assert features[2] == 1/8  # Exclamation mark ratio
    assert 0 < features[3] < 1  # Normalized length
    assert features[4] == 0  # Politeness ratio

if __name__ == "__main__":
    pytest.main()
