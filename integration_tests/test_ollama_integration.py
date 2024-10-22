import unittest
import json
from mas_dynamics_simulation.language_model.ollama_model import OllamaModel

class TestOllamaIntegration(unittest.TestCase):

    def setUp(self):
        self.model_params = {
            'base_url': 'http://localhost:11434',
            'model_name': 'llama3:instruct',
            'embedding_model': 'nomic-embed-text' #'mxbai-embed-large'
        }
        self.ollama_model = OllamaModel(self.model_params)
        

    def test_generate_text_integration(self):
        prompt = "What is the capital of France?"
        try:
            result = self.ollama_model.generate_text(prompt)
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 0)
        except json.JSONDecodeError as e:
            self.fail("JSONDecodeError occurred")

    def test_get_embedding_integration(self):
        text = "This is a test sentence."
        try:
            result = self.ollama_model.get_embedding(text)
            self.assertIsInstance(result, list)
            self.assertGreater(len(result), 0)
            self.assertIsInstance(result[0], float)
        except json.JSONDecodeError as e:
            self.fail("JSONDecodeError occurred")

if __name__ == '__main__':
    unittest.main()
