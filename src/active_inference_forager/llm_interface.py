from abc import ABC, abstractmethod

class LLMInterface(ABC):
    """Abstract base class for LLM interfaces."""

    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        """Generate a response to the given prompt."""
        pass

class MockLLM(LLMInterface):
    """A mock LLM for testing purposes."""

    def generate_response(self, prompt: str) -> str:
        """Generate a mock response to the given prompt."""
        return f"This is a mock response to: {prompt}"

# In the future, we can add more LLM implementations here, e.g.:
# class OpenAILLM(LLMInterface):
#     def __init__(self, api_key: str):
#         self.api_key = api_key
#
#     def generate_response(self, prompt: str) -> str:
#         # Implement OpenAI API call here
#         pass
