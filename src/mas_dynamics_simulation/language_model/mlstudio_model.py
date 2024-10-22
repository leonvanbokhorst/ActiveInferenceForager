import requests
from typing import Dict, Any, List
from .language_model import LanguageModel

class MLStudioModel(LanguageModel):
    """
    Concrete implementation of LanguageModel for the local MLStudio API.
    """

    def __init__(self, model_params: Dict[str, Any]):
        """
        Initialize the MLStudio model with given parameters.

        Args:
            model_params (Dict[str, Any]): Parameters for the MLStudio model.
                Should include 'base_url', 'model_name', and optionally 'api_key'.
        """
        super().__init__(model_params)
        self.base_url = model_params['base_url']
        self.model_name = model_params['model_name']
        self.api_key = model_params.get('api_key')
        self.headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

    def generate_text(self, prompt: str, max_tokens: int = 100) -> str:
        """
        Generate text using the MLStudio model based on the given prompt.

        Args:
            prompt (str): The input prompt for text generation.
            max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 100.

        Returns:
            str: The generated text.
        """
        url = f"{self.base_url}/v1/completions"
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": max_tokens
        }

        response = requests.post(url, json=data, headers=self.headers)
        if response.status_code == 200:
            return response.json()['choices'][0]['text'].strip()
        else:
            raise Exception(f"Error generating text: {response.status_code} - {response.text}")

    def get_embedding(self, text: str) -> List[float]:
        """
        Get the embedding representation of the given text using the MLStudio model.

        Args:
            text (str): The input text to embed.

        Returns:
            List[float]: The embedding vector.
        """
        url = f"{self.base_url}/v1/embeddings"
        data = {
            "model": self.model_name,
            "input": text
        }

        response = requests.post(url, json=data, headers=self.headers)
        if response.status_code == 200:
            return response.json()['data'][0]['embedding']
        else:
            raise Exception(f"Error getting embedding: {response.status_code} - {response.text}")
