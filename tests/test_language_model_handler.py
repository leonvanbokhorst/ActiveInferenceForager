import pytest
from unittest.mock import Mock, patch
from mas_dynamics_simulation.language_model import LanguageModel, language_model_handler
import re
from typing import Optional, Dict, Any, Type

class MockLanguageModel(LanguageModel):
    def __init__(self, model_params):
        self.model_params = model_params

    def generate_text(self, prompt: str, max_tokens: int = 100) -> str:
        return f"Generated: {prompt}"

    def get_embedding(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

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

@pytest.fixture
def reset_handler():
    # Reset the handler before and after each test
    language_model_handler._model = None
    yield
    language_model_handler._model = None

def test_initialize(reset_handler):
    model_params = {"param1": "value1", "param2": "value2"}
    language_model_handler.initialize(MockLanguageModel, model_params)
    assert isinstance(language_model_handler._model, MockLanguageModel)
    assert language_model_handler._model.model_params == model_params

def test_generate_text_without_initialization(reset_handler):
    error_message = re.escape("Language model not initialized. Call initialize() first.")
    with pytest.raises(RuntimeError, match=error_message):
        language_model_handler.generate_text("Test prompt")

def test_get_embedding_without_initialization(reset_handler):
    error_message = re.escape("Language model not initialized. Call initialize() first.")
    with pytest.raises(RuntimeError, match=error_message):
        language_model_handler.get_embedding("Test text")

def test_generate_text(reset_handler):
    language_model_handler.initialize(MockLanguageModel, {})
    result = language_model_handler.generate_text("Test prompt")
    assert result == "Generated: Test prompt"

def test_get_embedding(reset_handler):
    language_model_handler.initialize(MockLanguageModel, {})
    result = language_model_handler.get_embedding("Test text")
    assert result == [0.1, 0.2, 0.3]

def test_multiple_initializations(reset_handler):
    language_model_handler.initialize(MockLanguageModel, {"param": "value1"})
    first_model = language_model_handler._model
    
    language_model_handler.initialize(MockLanguageModel, {"param": "value2"})
    second_model = language_model_handler._model
    
    assert first_model is not second_model
    assert second_model.model_params["param"] == "value2"

@patch('mas_dynamics_simulation.language_model.language_model_handler._model')
def test_generate_text_calls_model(mock_model, reset_handler):
    mock_model.generate_text.return_value = "Mocked generated text"
    result = language_model_handler.generate_text("Test prompt", max_tokens=50)
    mock_model.generate_text.assert_called_once_with("Test prompt", max_tokens=50)
    assert result == "Mocked generated text"

@patch('mas_dynamics_simulation.language_model.language_model_handler._model')
def test_get_embedding_calls_model(mock_model, reset_handler):
    mock_model.get_embedding.return_value = [0.4, 0.5, 0.6]
    result = language_model_handler.get_embedding("Test text")
    mock_model.get_embedding.assert_called_once_with("Test text")
    assert result == [0.4, 0.5, 0.6]
