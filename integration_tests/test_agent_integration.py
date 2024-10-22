import unittest
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
        self.model_params = {
            'base_url': 'http://localhost:11434',
            'model_name': 'mistral-nemo'
        }
        self.language_model_handler = LanguageModelHandler()
        self.language_model_handler.initialize(OllamaModel, self.model_params)

    def test_agent_initialization(self):
        expertise = ["Human-machine interaction", "Bias and fairness in AI"]
        decision_engine = MockDecisionEngine()
        personality = BigFivePersonality.random()
        agent = ConcreteAgent(expertise, decision_engine, self.language_model_handler, personality)

        print(f"\nAgent Name: {agent.name}\n")
        print(f"Agent Backstory: \n{agent.backstory}\n")
        print(f"Agent Bio: \n{agent.bio}\n")
        print(f"Agent Dark Secret: \n{agent.dark_secret}\n")
        self.assertIsNotNone(agent.name)
        self.assertIsNotNone(agent.backstory)
        self.assertIsNotNone(agent.bio)
        self.assertEqual(agent.expertise, tuple(expertise))
        self.assertIsInstance(agent.decision_engine, MockDecisionEngine)
        self.assertIsInstance(agent.personality, BigFivePersonality)


if __name__ == '__main__':
    unittest.main()
