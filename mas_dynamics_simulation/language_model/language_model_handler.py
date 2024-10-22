from typing import Optional, Dict, Any, Type
from .language_model import LanguageModel

class LanguageModelHandler:
    def __init__(self):
        self.model = None

    def initialize(self, model_class: Type[LanguageModel], model_params: Dict[str, Any]):
        self.model = model_class(model_params)

    def generate_text(self, prompt: str, max_tokens: int = 100) -> str:
        if self.model is None:
            raise RuntimeError("Language model not initialized. Call initialize() first.")
        return self.model.generate_text(prompt, max_tokens=max_tokens)

    def get_embedding(self, text: str) -> list[float]:
        if self.model is None:
            raise RuntimeError("Language model not initialized. Call initialize() first.")
        return self.model.get_embedding(text)

    def reset(self):
        self.model = None

# Create a single instance of LanguageModelHandler to be used throughout the application
language_model_handler = LanguageModelHandler()

# Export the instance
__all__ = ['language_model_handler']
