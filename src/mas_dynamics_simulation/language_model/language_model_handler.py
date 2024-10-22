from typing import Optional, Dict, Any, Type
from .language_model import LanguageModel

_model: Optional[LanguageModel] = None

def initialize(model_class: Type[LanguageModel], model_params: Dict[str, Any]):
    global _model
    _model = model_class(model_params)

def generate_text(prompt: str, max_tokens: int = 100) -> str:
    if _model is None:
        raise RuntimeError("Language model not initialized. Call initialize() first.")
    return _model.generate_text(prompt, max_tokens=max_tokens)  # Use keyword argument here

def get_embedding(text: str) -> list[float]:
    if _model is None:
        raise RuntimeError("Language model not initialized. Call initialize() first.")
    return _model.get_embedding(text)

class LanguageModelHandler:
    def __init__(self):
        self._model = None

    def initialize(self, model_class: Type[LanguageModel], model_params: Dict[str, Any]):
        self._model = model_class(model_params)

    def generate_text(self, prompt: str, max_tokens: int = 100) -> str:
        if self._model is None:
            raise RuntimeError("Language model not initialized. Call initialize() first.")
        return self._model.generate_text(prompt, max_tokens=max_tokens)

    def get_embedding(self, text: str) -> list[float]:
        if self._model is None:
            raise RuntimeError("Language model not initialized. Call initialize() first.")
        return self._model.get_embedding(text)
