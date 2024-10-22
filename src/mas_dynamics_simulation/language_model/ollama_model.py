import ollama
from typing import Dict, Any, List
from .language_model import LanguageModel

class OllamaModel(LanguageModel):
    """
    Concrete implementation of LanguageModel for the local Ollama server.
    """

    def __init__(self, model_params: Dict[str, Any]):
        """
        Initialize the Ollama model with given parameters.

        Args:
            model_params (Dict[str, Any]): Parameters for the Ollama model.
                Should include 'base_url' and 'model_name'.
        """
        super().__init__(model_params)
        self.model_name = model_params['model_name']
        self.embedding_model = model_params.get('embedding_model', self.model_name)

    def generate_text(self, prompt: str, max_tokens: int = 100) -> str:
        """
        Generate text using the Ollama model based on the given prompt.

        Args:
            prompt (str): The input prompt for text generation.
            max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 100.

        Returns:
            str: The generated text.
        """
        response = ollama.generate(model=self.model_name, prompt=prompt)
        return response['response']

    def get_embedding(self, text: str) -> List[float]:
        """
        Get the embedding representation of the given text using the Ollama model.

        Args:
            text (str): The input text to embed.

        Returns:
            List[float]: The embedding vector.
        """
        response = ollama.embeddings(model=self.embedding_model, prompt=text)
        return response['embedding']
