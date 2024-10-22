from typing import Optional, Dict, Any, Type
from .language_model import LanguageModel

class LanguageModelHandler:
    def __init__(self):
        self.model: Optional['LanguageModel'] = None

    @property
    def is_initialized(self) -> bool:
        """Check if the language model is initialized."""
        return self.model is not None

    def initialize(self, model_class: Type['LanguageModel'], model_params: Dict[str, Any]) -> None:
        """Initialize the language model with the given class and parameters."""
        self.model = model_class(model_params)

    def generate_text(self, prompt: str, max_tokens: int = 100) -> str:
        """Generate text based on the given prompt."""
        self._check_initialization()
        return self.model.generate_text(prompt, max_tokens=max_tokens)

    def get_embedding(self, text: str) -> list[float]:
        """Get the embedding for the given text."""
        self._check_initialization()
        return self.model.get_embedding(text)

    def reset(self) -> None:
        """Reset the language model handler."""
        self.model = None

    def _check_initialization(self) -> None:
        """Check if the model is initialized and raise an error if not."""
        if not self.is_initialized:
            raise RuntimeError("Language model not initialized. Call initialize() first.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.reset()

