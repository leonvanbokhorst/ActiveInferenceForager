import pytest
from active_inference_forager.active_inference_agent import ActiveInferenceAgent
from active_inference_forager.llm_interface import MockLLM

@pytest.fixture
def active_inference_agent():
    llm = MockLLM()
    return ActiveInferenceAgent(llm)

class TestActiveInferenceAgent:
    def test_process_user_input(self, active_inference_agent):
        response = active_inference_agent.process_user_input("Hello")
        assert isinstance(response, str)
        assert len(response) > 0
