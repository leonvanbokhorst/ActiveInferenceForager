import unittest
import logging
from mas_dynamics_simulation.agent import Agent, Action
from mas_dynamics_simulation.language_model.ollama_model import OllamaModel
from mas_dynamics_simulation.language_model.language_model_handler import LanguageModelHandler
from mas_dynamics_simulation.decision_making import DecisionEngine
from mas_dynamics_simulation.personality import BigFivePersonality, Personality
from mas_dynamics_simulation.environment import Environment
from typing import List, Dict

class MockDecisionEngine(DecisionEngine):
    def evaluate_options(self, agent, options, environment):
        return options[0] if options else None

    def __str__(self):
        return "MockDecisionEngine"

class MockEnvironment(Environment):
    def get_state(self):
        return {}

    def apply_action(self, agent, action):
        pass

    def update(self):
        pass

class ConcreteAgent(Agent):
    @classmethod
    def generate_agent_details(cls, expertise: List[str], personality: Personality, decision_engine: "DecisionEngine", language_model_handler: LanguageModelHandler) -> Dict[str, str]:
        return super().generate_agent_details(expertise, personality, decision_engine, language_model_handler)

    def perceive(self, environment: Environment):
        return {}

    def decide(self, perception):
        return Action()

    def act(self, action, environment):
        pass

    def update(self, feedback):
        pass

    def __str__(self):
        return f"ConcreteAgent: {self.name}"

class Action:
    def __init__(self):
        self.name = "MockAction"

    def execute(self, agent, environment):
        pass

    def __str__(self):
        return self.name

class TestAgentIntegration(unittest.TestCase):
    def setUp(self):
        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        self.model_params = {
            'base_url': 'http://localhost:11434',
            'model_name': 'mistral-nemo'
        }
        self.language_model_handler = LanguageModelHandler()
        self.language_model_handler.initialize(OllamaModel, self.model_params)

    def test_agent_initialization(self):
        expertise = ["Project Management", "Leadership", "Teamwork", "Communication", "Problem Solving", "Adaptability", "Innovation", "Critical Thinking"]
        decision_engine = MockDecisionEngine()
        personality = BigFivePersonality.random()
        agent = ConcreteAgent(expertise, decision_engine, self.language_model_handler, personality)

        self.assertIsNotNone(agent.name)
        self.assertIsNotNone(agent.backstory)
        self.assertIsNotNone(agent.bio)
        self.assertIsNotNone(agent.dark_secret)
        self.assertEqual(agent.expertise, tuple(expertise))
        self.assertIsInstance(agent.decision_engine, MockDecisionEngine)
        self.assertIsInstance(agent.personality, BigFivePersonality)
        
        self.logger.info(f"Agent Name: {agent.name}")
        self.logger.info(f"Agent Backstory: {agent.backstory}")
        self.logger.info(f"Agent Bio: {agent.bio}")
        self.logger.info(f"Agent Dark Secret: {agent.dark_secret}")


if __name__ == '__main__':
    unittest.main()
