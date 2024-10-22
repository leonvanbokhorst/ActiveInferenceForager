import unittest
import json
from mas_dynamics_simulation.language_model.mlstudio_model import MLStudioModel

class TestMLStudioIntegration(unittest.TestCase):

    def setUp(self):
        self.model_params = {
            'base_url': 'http://localhost:1234',  # Adjust this to your MLStudio API endpoint
            'model_name': 'meta-llama-3.1-8b-instruct-4bit',  # Adjust this to the model name you're using
            'api_key': 'LMSTUDIO',
            'embedding_model': 'nomic-embed-text',
            'max_tokens': 100,
            'temperature': 0.7,
            'top_p': 0.95,
            'frequency_penalty': 0.0,
            'presence_penalty': 0.0
        }
        self.mlstudio_model = MLStudioModel(self.model_params)

    def test_generate_text_integration(self):
        prompt = "What is the capital of France?"
        try:
            result = self.mlstudio_model.generate_text(prompt)
            print(f"Generated text: {result}")
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 0)
        except json.JSONDecodeError as e:
            self.fail(f"JSONDecodeError occurred: {str(e)}")
        except Exception as e:
            self.fail(f"An error occurred: {str(e)}")

    def test_get_embedding_integration(self):
        text = "This is a test sentence."
        try:
            result = self.mlstudio_model.get_embedding(text)
            print(f"Embedding: {result}")
            self.assertIsInstance(result, list)
            self.assertGreater(len(result), 0)
            self.assertIsInstance(result[0], float)
        except json.JSONDecodeError as e:
            self.fail(f"JSONDecodeError occurred: {str(e)}")
        except Exception as e:
            self.fail(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    unittest.main()
