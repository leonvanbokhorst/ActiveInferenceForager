import pytest
from unittest.mock import Mock
from mas_dynamics_simulation.language_model.language_model_handler import LanguageModelHandler

class MockLanguageModel:
    def __init__(self, params):
        self.params = params

    def generate_text(self, prompt, max_tokens):
        return f"Generated: {prompt[:10]}..."

    def get_embedding(self, text):
        return [0.1, 0.2, 0.3]

@pytest.fixture
def handler():
    return LanguageModelHandler()

def test_is_initialized(handler):
    assert not handler.is_initialized
    handler.initialize(MockLanguageModel, {"param": "value"})
    assert handler.is_initialized

def test_generate_text(handler):
    handler.initialize(MockLanguageModel, {})
    result = handler.generate_text("Test prompt", max_tokens=10)
    assert result == "Generated: Test promp..."

def test_generate_text_uninitialized(handler):
    with pytest.raises(RuntimeError):
        handler.generate_text("Test prompt")

def test_get_embedding(handler):
    handler.initialize(MockLanguageModel, {})
    result = handler.get_embedding("Test text")
    assert result == [0.1, 0.2, 0.3]

def test_get_embedding_uninitialized(handler):
    with pytest.raises(RuntimeError):
        handler.get_embedding("Test text")

def test_reset(handler):
    handler.initialize(MockLanguageModel, {})
    assert handler.is_initialized
    handler.reset()
    assert not handler.is_initialized

def test_context_manager():
    with LanguageModelHandler() as handler:
        handler.initialize(MockLanguageModel, {})
        assert handler.is_initialized
    assert not handler.is_initialized

def test_initialize(handler):
    assert not handler.is_initialized
    handler.initialize(MockLanguageModel, {"param": "value"})
    assert handler.is_initialized
    assert isinstance(handler.model, MockLanguageModel)
    assert handler.model.params == {"param": "value"}
