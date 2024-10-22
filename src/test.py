import sys
import os
import math

from typing import Dict, Any, List
from mas_dynamics_simulation.agent import Agent, Action
from mas_dynamics_simulation.personality import Personality
from mas_dynamics_simulation.decision_making import DecisionEngine
from mas_dynamics_simulation.environment import Environment

class SimpleAgent(Agent):
    def __init__(self, name: str, personality: Personality, expertise: List[str], decision_engine: DecisionEngine):
        super().__init__(name, personality, expertise, decision_engine)

    def perceive(self, environment: Environment) -> Dict[str, Any]:
        # Simple perception: just return the agent's name and current time
        return {"agent_name": self.name, "current_time": environment.current_time}

    def decide(self, perception: Dict[str, Any]) -> Action:
        # Simple decision: always return a "DoNothing" action
        return DoNothingAction()

    def act(self, action: Action, environment: Environment):
        # Simply execute the action
        action.execute(self, environment)

    def update(self, feedback: Dict[str, Any]):
        # Simple update: print the feedback
        print(f"Agent {self.name} received feedback: {feedback}")

    def __str__(self) -> str:
        return f"SimpleAgent(name={self.name}, expertise={self.expertise})"


class DoNothingAction(Action):
    @property
    def name(self) -> str:
        return "DoNothing"

    def execute(self, agent: Agent, environment: Environment):
        print(f"Agent {agent.name} is doing nothing.")

    def __str__(self) -> str:
        return "DoNothingAction()"

class SimpleEnvironment(Environment):
    def __init__(self):
        super().__init__()
        self.current_time = 0
        self.state = {}  # Add a state dictionary to store environment state

    def step(self):
        self.current_time += 1

    def apply_action(self, agent: Agent, action: Action):
        # Implement the logic to apply an action to the environment
        print(f"Applying action {action} from agent {agent.name}")
        # You can update the environment state here based on the action

    def get_state(self) -> Dict[str, Any]:
        # Return the current state of the environment
        return {
            "current_time": self.current_time,
            **self.state  # Include any other state variables
        }

    def update(self):
        # Update the environment state
        # This method can be used to apply any time-based changes or events
        print(f"Updating environment at time {self.current_time}")

    # Add any other necessary methods

class SimpleSimulation:
    def __init__(self):
        self.environment = None
        self.agents = []

    def initialize(self, environment, agents):
        self.environment = environment
        self.agents = agents

    def run(self):
        print("Running simulation...")
        for _ in range(5):  # Run for 5 steps
            for agent in self.agents:
                perception = agent.perceive(self.environment)
                action = agent.decide(perception)
                self.environment.apply_action(agent, action)
                agent.act(action, self.environment)
            self.environment.update()
            self.environment.step()
            print(f"Environment state: {self.environment.get_state()}")

class SimpleDecisionEngine(DecisionEngine):
    def __init__(self):
        super().__init__()

    def __str__(self) -> str:
        return "SimpleDecisionEngine()"

    def make_decision(self, perception: Dict[str, Any]) -> Action:
        return DoNothingAction()

    def evaluate_options(self, options: List[Action], perception: Dict[str, Any]) -> Dict[Action, float]:
        # Simple evaluation: assign equal probability to all options
        num_options = len(options)
        if num_options == 0:
            return {}
        probability = 1.0 / num_options
        return {action: probability for action in options}

class SimplePersonality(Personality):
    def __init__(self, traits: Dict[str, float] = None):
        traits = traits or {"openness": 0.5, "conscientiousness": 0.5, "extraversion": 0.5, "agreeableness": 0.5, "neuroticism": 0.5}
        super().__init__(traits)
        self.traits = traits

    def __str__(self) -> str:
        return f"SimplePersonality(traits={self.traits})"

    def similarity(self, other: 'Personality') -> float:
        if not isinstance(other, SimplePersonality):
            return 0.0
        return self.cosine_similarity(self.traits, other.traits)

    @staticmethod
    def cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        common_keys = set(vec1.keys()) & set(vec2.keys())
        
        dot_product = sum(vec1[k] * vec2[k] for k in common_keys)
        magnitude1 = math.sqrt(sum(v**2 for v in vec1.values()))
        magnitude2 = math.sqrt(sum(v**2 for v in vec2.values()))
        
        if magnitude1 * magnitude2 == 0:
            return 0.0
        return dot_product / (magnitude1 * magnitude2)

    # Add any other necessary methods

if __name__ == "__main__":
    # Create a simple agent with a custom personality
    personality = SimplePersonality({"openness": 0.7, "conscientiousness": 0.6, "extraversion": 0.5, "agreeableness": 0.8, "neuroticism": 0.3})
    decision_engine = SimpleDecisionEngine()
    agent = SimpleAgent(name="Agent1", personality=personality, expertise=["Expertise1"], decision_engine=decision_engine)
    agent2 = SimpleAgent(name="Agent2", personality=personality, expertise=["Expertise2"], decision_engine=decision_engine)

    # Create the environment
    environment = SimpleEnvironment()

    # Run the simulation
    simulation = SimpleSimulation()
    simulation.initialize(environment, [agent, agent2])
    simulation.run()
