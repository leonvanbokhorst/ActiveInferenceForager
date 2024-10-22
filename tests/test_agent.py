import pytest
import json
from unittest.mock import Mock, patch
from mas_dynamics_simulation.agent import Agent
from mas_dynamics_simulation.decision_making import DecisionEngine
from mas_dynamics_simulation.personality import Personality, DefaultPersonality
from mas_dynamics_simulation.language_model.language_model_handler import LanguageModelHandler

@pytest.fixture
def mock_decision_engine():
    return Mock(spec=DecisionEngine)

@pytest.fixture
def mock_language_model_handler():
    mock = Mock(spec=LanguageModelHandler)
    mock.generate_text.return_value = json.dumps({
        "name": "Test Agent",
        "backstory": "A test agent's backstory",
        "bio": "A short bio for the test agent"
    })
    return mock

@pytest.fixture
def mock_personality():
    return Mock(spec=Personality)

@pytest.fixture
def default_personality():
    return DefaultPersonality()

@pytest.fixture
def test_agent(mock_decision_engine, mock_language_model_handler, mock_personality):
    class ConcreteAgent(Agent):
        def perceive(self, environment):
            pass

        def decide(self, perception):
            pass

        def act(self, action, environment):
            pass

        def update(self, feedback):
            pass

        def __str__(self):
            return f"Agent: {self.name}"

    return ConcreteAgent(
        expertise=['testing'],
        decision_engine=mock_decision_engine,
        language_model_handler=mock_language_model_handler,
        personality=mock_personality
    )

def test_init(test_agent, mock_language_model_handler):
    assert isinstance(test_agent.name, str)
    assert isinstance(test_agent.backstory, str)
    assert isinstance(test_agent.bio, str)
    assert test_agent.expertise == ('testing',)
    assert isinstance(test_agent.decision_engine, DecisionEngine)
    assert isinstance(test_agent._language_model_handler, LanguageModelHandler)  # Changed from language_model_handler to _language_model_handler
    assert isinstance(test_agent.personality, Personality)
    
    # Verify that the language model was called to generate agent details
    mock_language_model_handler.generate_text.assert_called_once()


def test_think(test_agent, mock_language_model_handler):
    # Reset the mock to clear the call count from initialization
    mock_language_model_handler.generate_text.reset_mock()
    
    mock_language_model_handler.generate_text.return_value = "I think this is a good idea."
    result = test_agent.think({"situation": "new project"})
    assert result == "I think this is a good idea."
    mock_language_model_handler.generate_text.assert_called_once()

    # Verify the content of the call
    call_args = mock_language_model_handler.generate_text.call_args[0][0]
    assert "As Test Agent, an expert in testing, think about the following context:" in call_args
    assert "{'situation': 'new project'}" in call_args

def test_memorize_and_remember(test_agent, mock_language_model_handler):
    mock_language_model_handler.generate_text.reset_mock()
    test_agent.memorize("Important information")
    mock_language_model_handler.generate_text.return_value = "Recalled: Important information"
    result = test_agent.remember("What's important?")
    assert result == "Recalled: Important information"
    mock_language_model_handler.generate_text.assert_called_once()

def test_talk(test_agent, mock_language_model_handler):
    mock_language_model_handler.generate_text.reset_mock()
    mock_language_model_handler.generate_text.return_value = "Hello, I'm the test agent."
    result = test_agent.talk("Introduce yourself")
    assert result == "Hello, I'm the test agent."
    mock_language_model_handler.generate_text.assert_called_once()
