from abc import ABC, abstractmethod
from typing import Dict, Any, List

class LanguageModel(ABC):
    """
    Abstract base class for language models used in the multi-agent simulation.
    """

    @abstractmethod
    def __init__(self, model_params: Dict[str, Any]):
        """
        Initialize the language model with given parameters.

        Args:
            model_params (Dict[str, Any]): Parameters for the language model.
        """
        pass

    @abstractmethod
    def generate_text(self, prompt: str, max_tokens: int = 100) -> str:
        """
        Generate text based on the given prompt.

        Args:
            prompt (str): The input prompt for text generation.
            max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 100.

        Returns:
            str: The generated text.
        """
        pass

    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        """
        Get the embedding representation of the given text.

        Args:
            text (str): The input text to embed.

        Returns:
            List[float]: The embedding vector.
        """
        pass
