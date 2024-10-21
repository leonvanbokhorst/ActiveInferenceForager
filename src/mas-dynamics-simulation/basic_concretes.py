import random
from typing import List, Dict, Any
from abc import ABC, abstractmethod

from abstract_classes import (
    AbstractEnvironment,
    AbstractAgent,
    AbstractDecisionEngine,
    WorldView,
    Goal,
)


class ConcreteEnvironment(AbstractEnvironment):
    def __init__(self, size: int):
        self.size = size
        self.time = 0
        self.resources = {
            (x, y): random.randint(0, 100) for x in range(size) for y in range(size)
        }
        self.agents: Dict[str, ConcreteAgent] = {}

    def get_state(self) -> Dict[str, Any]:
        return {
            "time": self.time,
            "resources": self.resources,
            "agents": {
                agent_id: agent.get_position()
                for agent_id, agent in self.agents.items()
            },
        }

    def apply_action(self, action: str, agent: "ConcreteAgent"):
        if action == "move":
            new_pos = self._get_random_adjacent_position(agent.get_position())
            agent.set_position(new_pos)
        elif action == "gather":
            resources = self.resources.get(agent.get_position(), 0)
            gathered = min(resources, agent.gather_capacity)
            self.resources[agent.get_position()] -= gathered
            agent.resources += gathered

    def generate_event(self) -> str:
        events = ["rain", "sunshine", "drought", "abundance"]
        return random.choice(events)

    def advance_time(self, time_step: int):
        self.time += time_step

    def update(self, world_view: WorldView):
        world_view.update_knowledge("environment_state", self.get_state())

    def _get_random_adjacent_position(self, position: tuple) -> tuple:
        x, y = position
        dx, dy = random.choice([(0, 1), (1, 0), (0, -1), (-1, 0)])
        return ((x + dx) % self.size, (y + dy) % self.size)


class ConcreteAgent(AbstractAgent):
    def __init__(self, agent_id: str, world_view: WorldView, position: tuple):
        super().__init__(agent_id, world_view)
        self.position = position
        self.resources = 0
        self.gather_capacity = 10
        self.decision_engine = ConcreteDecisionEngine()

    def perceive_environment(self, environment: ConcreteEnvironment):
        visible_area = self._get_visible_area(environment)
        self.world_view.update_knowledge("visible_area", visible_area)

    def update_mental_model(self, agent: "ConcreteAgent", interaction: str):
        if agent.agent_id not in self.mental_models:
            self.mental_models[agent.agent_id] = AgentModel(agent.agent_id)
        self.mental_models[agent.agent_id].update_model(interaction)

    def get_position(self) -> tuple:
        return self.position

    def set_position(self, position: tuple):
        self.position = position

    def _get_visible_area(self, environment: ConcreteEnvironment) -> Dict[tuple, int]:
        x, y = self.position
        visible_area = {}
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                pos = ((x + dx) % environment.size, (y + dy) % environment.size)
                visible_area[pos] = environment.resources.get(pos, 0)
        return visible_area

    def update_emotional_state(self, event: str):
        # Simplified emotional state update
        pass

    def evaluate_relationship(self, other_agent: "ConcreteAgent") -> float:
        # Simplified relationship evaluation
        return random.random()


class ConcreteDecisionEngine(AbstractDecisionEngine):
    def make_decision(self, agent: ConcreteAgent, options: List[Any]) -> Any:
        # Simple decision-making: choose the option with the highest expected utility
        return max(options, key=lambda option: self.evaluate_option(agent, option))

    def evaluate_option(self, agent: ConcreteAgent, option: Any) -> float:
        if option == "move":
            return 0.5  # Base utility for moving
        elif option == "gather":
            visible_area = agent.world_view.get_knowledge("visible_area")
            resources_at_position = visible_area.get(agent.get_position(), 0)
            return (
                min(resources_at_position, agent.gather_capacity)
                / agent.gather_capacity
            )
        return 0  # Default utility for unknown options

    def prioritize_goals(self, agent: ConcreteAgent, goals: List[Goal]) -> List[Goal]:
        # Simple prioritization: sort goals by their priority attribute
        return sorted(goals, key=lambda g: g.priority, reverse=True)


# Example usage
if __name__ == "__main__":
    world_view = WorldView({})
    environment = ConcreteEnvironment(size=25)
    agent = ConcreteAgent("Agent1", world_view, position=(0, 0))
    agent2 = ConcreteAgent("Agent2", world_view, position=(5, 21))
    environment.agents[agent.agent_id] = agent
    environment.agents[agent2.agent_id] = agent2

    for _ in range(5):  # Simulate 5 steps
        agent.perceive_environment(environment)
        options = ["move", "gather"]
        action = agent.make_decision(options)
        environment.apply_action(action, agent)
        environment.advance_time(1)
        print(
            f"Step {environment.time}: Agent at {agent.get_position()} with {agent.resources} resources"
        )

    print("Final environment state:", environment.get_state())
