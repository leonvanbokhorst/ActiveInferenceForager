import pytest
import requests
from unittest.mock import patch, MagicMock
from mas_dynamics_simulation.language_model.mlstudio_model import MLStudioModel

@pytest.fixture
def model_params():
    return {
        'base_url': 'http://example.com/api',
        'model_name': 'test-model',
        'api_key': 'test-api-key'
    }

@pytest.fixture
def mlstudio_model(model_params):
    return MLStudioModel(model_params)

def test_init(model_params):
    model = MLStudioModel(model_params)
    assert model.base_url == model_params['base_url']
    assert model.model_name == model_params['model_name']
    assert model.api_key == model_params['api_key']
    assert model.headers == {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {model_params['api_key']}"
    }

@patch('requests.post')
def test_generate_text_success(mock_post, mlstudio_model):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {'choices': [{'text': ' Generated text'}]}
    mock_post.return_value = mock_response

    result = mlstudio_model.generate_text("Test prompt", max_tokens=50)

    assert result == "Generated text"
    mock_post.assert_called_once_with(
        f"{mlstudio_model.base_url}/v1/completions",
        json={"model": mlstudio_model.model_name, "prompt": "Test prompt", "max_tokens": 50},
        headers=mlstudio_model.headers
    )

@patch('requests.post')
def test_generate_text_failure(mock_post, mlstudio_model):
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.text = "Bad Request"
    mock_post.return_value = mock_response

    with pytest.raises(Exception) as exc_info:
        mlstudio_model.generate_text("Test prompt")

    assert str(exc_info.value) == "Error generating text: 400 - Bad Request"

@patch('requests.post')
def test_get_embedding_success(mock_post, mlstudio_model):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {'data': [{'embedding': [0.1, 0.2, 0.3]}]}
    mock_post.return_value = mock_response

    result = mlstudio_model.get_embedding("Test text")

    assert result == [0.1, 0.2, 0.3]
    mock_post.assert_called_once_with(
        f"{mlstudio_model.base_url}/v1/embeddings",
        json={"model": mlstudio_model.model_name, "input": "Test text"},
        headers=mlstudio_model.headers
    )

@patch('requests.post')
def test_get_embedding_failure(mock_post, mlstudio_model):
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    mock_post.return_value = mock_response

    with pytest.raises(Exception) as exc_info:
        mlstudio_model.get_embedding("Test text")

    assert str(exc_info.value) == "Error getting embedding: 500 - Internal Server Error"
