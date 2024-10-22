import unittest
from unittest.mock import patch, Mock
from mas_dynamics_simulation.language_model.ollama_model import OllamaModel

class TestOllamaModel(unittest.TestCase):

    def setUp(self):
        self.model_params = {
            'model_name': 'llama3.1-mock',
            'embedding_model': 'llama3.1-mock-embed'
        }
        self.ollama_model = OllamaModel(self.model_params)

    @patch('mas_dynamics_simulation.language_model.ollama_model.ollama.generate')
    def test_generate_text(self, mock_generate):
        # Mock the response from ollama.generate
        mock_generate.return_value = {
            'response': 'Generated text'
        }

        result = self.ollama_model.generate_text("Test prompt")
        self.assertEqual(result, 'Generated text')

        mock_generate.assert_called_once_with(model='llama3.1-mock', prompt='Test prompt')

    @patch('mas_dynamics_simulation.language_model.ollama_model.ollama.embeddings')
    def test_get_embedding(self, mock_embeddings):
        # Mock the response from ollama.embeddings
        mock_embeddings.return_value = {
            'embedding': [0.1, 0.2, 0.3]
        }

        result = self.ollama_model.get_embedding("Test text")
        self.assertEqual(result, [0.1, 0.2, 0.3])

        mock_embeddings.assert_called_once_with(model='llama3.1-mock-embed', prompt='Test text')

if __name__ == '__main__':
    unittest.main()
