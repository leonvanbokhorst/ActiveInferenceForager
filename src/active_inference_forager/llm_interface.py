from abc import ABC, abstractmethod
from typing import Dict, List


class LLMInterface(ABC):

    @abstractmethod
    def generate_response(self, prompt: str, context: Dict[str, any]) -> str:
        pass


class MockLLM(LLMInterface):

    def generate_response(self, prompt: str, context: Dict[str, any]) -> str:
        predicted_topic = context.get("predicted_topic", "general")
        confidence = context.get("confidence", 0.5)

        if confidence < 0.6:
            return f"I'm not quite sure, but I think we're talking about {predicted_topic}. Can you tell me more about what you're interested in?"
        else:
            return f"Based on our conversation about {predicted_topic}, I think the answer to '{prompt}' is: This is a mock response considering the context."


# In the future, we can add more sophisticated LLM implementations here, e.g.:
# class OpenAILLM(LLMInterface):
#     def __init__(self, api_key: str):
#         self.api_key = api_key
#
#     def generate_response(self, prompt: str, context: Dict[str, any]) -> str:
#         # Implement OpenAI API call here, using the context to enhance the prompt
#         pass
