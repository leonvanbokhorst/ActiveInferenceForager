import unittest
import numpy as np
from src.active_inference_forager.main import (
    initialize_state,
    update_intent_probabilities,
    calculate_surprise,
    predict_intent,
    process_input,
    INTENTS
)

class TestFEPConversation(unittest.TestCase):
    def test_initialize_state(self):
        state = initialize_state()
        self.assertIn('intent_probs', state)
        self.assertIn('last_input', state)
        self.assertIn('surprise_history', state)
        self.assertEqual(len(state['intent_probs']), len(INTENTS))
        self.assertAlmostEqual(np.sum(state['intent_probs']), 1.0)

    def test_update_intent_probabilities(self):
        probs = np.array([0.25, 0.25, 0.25, 0.25])
        updated_probs = update_intent_probabilities(probs, 0.5)
        self.assertAlmostEqual(np.sum(updated_probs), 1.0)
        self.assertTrue(np.all(updated_probs > 0))

    def test_calculate_surprise(self):
        self.assertEqual(calculate_surprise('greeting', 'greeting'), 0)
        self.assertEqual(calculate_surprise('greeting', 'question'), 1)

    def test_predict_intent(self):
        self.assertEqual(predict_intent('Hello!'), 'greeting')
        self.assertEqual(predict_intent('What is the weather like?'), 'question')
        self.assertEqual(predict_intent('I like pizza.'), 'statement')
        self.assertEqual(predict_intent('Goodbye!'), 'farewell')

    def test_process_input(self):
        state = initialize_state()
        new_state = process_input('Hello!', state)
        self.assertEqual(new_state['last_input'], 'Hello!')
        self.assertEqual(new_state['last_intent'], 'greeting')
        self.assertGreater(len(new_state['surprise_history']), 0)

if __name__ == '__main__':
    unittest.main()
