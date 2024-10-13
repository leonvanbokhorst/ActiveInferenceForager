import pytest
from active_inference_forager.conversation_manager import ConversationManager
from active_inference_forager.user_model import UserModel
from active_inference_forager.fep_predictor import FEPPredictor
from active_inference_forager.active_inference_agent import ActiveInferenceAgent
from active_inference_forager.llm_interface import MockLLM

@pytest.fixture
def conversation_manager():
    return ConversationManager()

@pytest.fixture
def user_model():
    return UserModel()

@pytest.fixture
def fep_predictor(user_model):
    return FEPPredictor(user_model)

@pytest.fixture
def active_inference_agent():
    llm = MockLLM()
    return ActiveInferenceAgent(llm)

def test_integration(conversation_manager, user_model, fep_predictor, active_inference_agent):
    # Test the integration of all components
    user_input = "Hello, I'm interested in AI"
    conversation_manager.add_message("user", user_input)
    
    # Process user input
    response = active_inference_agent.process_user_input(user_input)
    conversation_manager.add_message("agent", response)
    
    # Update user model
    user_model.update_preference("AI", 0.8)
    
    # Predict next topic
    next_topic = fep_predictor.predict_next_topic()
    
    # Assertions
    assert len(conversation_manager.get_conversation_history()) == 2
    assert isinstance(response, str)
    assert len(response) > 0
    assert user_model.get_preferences()["AI"] == 0.8
    assert next_topic == "AI"
