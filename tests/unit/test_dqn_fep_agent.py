import pytest
import numpy as np
import torch
from active_inference_forager.agents.generic_agent import GenericAgent, ExperienceReplayBuffer, DQN


@pytest.fixture
def agent():
    action_space = [
        "ask_question",
        "provide_information",
        "clarify",
        "suggest_action",
        "express_empathy",
        "end_conversation",
    ]
@pytest.fixture
def agent():
    action_space = [
        "express_empathy",
        "end_conversation",
    ]
    agent = GenericAgent(action_dim=len(action_space))
    assert agent.state_dim == 17
    return agent


def test_agent_initialization(agent):
    assert isinstance(agent, GenericAgent)
    assert agent.state_dim == 17
    assert agent.action_space == [
        "ask_question",
        "provide_information",
        "clarify",
        "suggest_action",
        "express_empathy",
        "end_conversation",
    ]
    assert isinstance(agent.q_network, DQN)
    assert isinstance(agent.target_network, DQN)
    assert isinstance(agent.optimizer, torch.optim.Adam)
    assert agent.exploration_rate == agent.epsilon_start


def test_take_action(agent):
    state = np.random.rand(17)
    action = agent.take_action(state)
    assert action in agent.action_space


def test_learn(agent):
    state = np.random.rand(17)
    action = "ask_question"
    next_state = np.random.rand(17)
    reward = 1.0
    done = False

    initial_total_steps = agent.total_steps

    agent.learn(state, action, next_state, reward, done)

    assert agent.total_steps > initial_total_steps


def test_update_belief(agent):
    observation = np.random.rand(17)
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
    action = "ask_question"
    interpretation = agent.interpret_action(action)
    assert isinstance(interpretation, str)
    assert "ask a question" in interpretation.lower()


def test_process_user_input(agent):
    user_input = "Hello, how are you? I'm feeling great today!"
    features = agent.process_user_input(user_input)
    assert isinstance(features, np.ndarray)
    assert features.shape == (17,)
    assert features[0] == 8  # Word count
    assert features[1] > 0  # Average word length
    assert features[2] == 1 / 8  # Question mark frequency
    assert features[3] == 1 / 8  # Exclamation mark frequency
    assert -1 <= features[4] <= 1  # Sentiment polarity
    assert 0 <= features[5] <= 1  # Subjectivity
    assert 0 <= features[6] <= 1  # Keyword detection
    assert 0 <= features[7] <= 1  # Lexical diversity
    assert 0 <= features[8] <= 1  # Proportion of long words
    assert 0 <= features[9] <= 1  # Politeness indicator
    assert 0 <= features[10] <= 1  # Named entity density
    assert 0 <= features[11] <= 1  # Noun density
    assert 0 <= features[12] <= 1  # Verb density
    assert 0 <= features[13] <= 1  # Main clause density
    assert features[14] > 0  # Average parse tree depth
    assert 0 <= features[15] <= 1  # Stop word density
    assert 0 <= features[16] <= 1  # Punctuation density


if __name__ == "__main__":
    pytest.main()
