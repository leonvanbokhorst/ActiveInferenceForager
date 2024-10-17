from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    def generate_response(self, prompt, **kwargs):
        """Generate a response from the LLM based on the given prompt."""
        pass
