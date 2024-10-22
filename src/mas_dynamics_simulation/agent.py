from abc import ABC, abstractmethod
from typing import Dict, Any, TYPE_CHECKING, List
import numpy as np
from mas_dynamics_simulation.language_model.language_model_handler import LanguageModelHandler
from mas_dynamics_simulation.decision_making import DecisionEngine
from mas_dynamics_simulation.personality import Personality

if TYPE_CHECKING:
    from .environment import Environment
    from .personality import Personality
    from .decision_making import DecisionEngine


class Agent(ABC):
    """
    Represents an autonomous agent in the multi-agent system. 
    Manages the agent's state, personality, expertise, and decision-making process.
    """

    def __init__(self, expertise: List[str], decision_engine: "DecisionEngine", language_model_handler: LanguageModelHandler, personality: Personality):
        """
        Initializes the agent with generated details based on expertise and decision engine.

        Args:
            expertise: The expertise of the agent.
            decision_engine: The decision-making engine of the agent.
            language_model_handler: The language model handler for text generation.
            personality: The personality of the agent.
        """
        self._language_model_handler = language_model_handler
        details = self.generate_agent_details(expertise, decision_engine, language_model_handler)
        self._name = details.get('name', 'Unknown Agent')
        self._backstory = details.get('backstory', 'No backstory available')
        self._bio = details.get('bio', 'No bio available')
        self._expertise = tuple(expertise)  # Make expertise immutable
        self._decision_engine = decision_engine
        self._personality = personality
        self._memory = []  # Initialize an empty memory

    @classmethod
    def generate_agent_details(cls, expertise: List[str], decision_engine: "DecisionEngine", language_model_handler: LanguageModelHandler) -> Dict[str, str]:
        """
        Generate agent details using the language model based on expertise and decision engine.

        Args:
            expertise: The expertise of the agent.
            decision_engine: The decision-making engine of the agent.
            language_model_handler: The language model handler for text generation.

        Returns:
            Dict[str, str]: A dictionary containing the generated agent details.
        """
        prompt = f"""
        Generate a name, backstory, and short bio for an AI agent with the following expertise: {', '.join(expertise)}.
        Format the response as a JSON object with keys 'name', 'backstory', and 'bio'.
        """
        response = language_model_handler.generate_text(prompt)
        try:
            import json
            details = json.loads(response)
            return details
        except json.JSONDecodeError:
            # If JSON parsing fails, return a default dictionary
            return {
                'name': 'Default Agent',
                'backstory': 'A mysterious agent with unknown origins.',
                'bio': f'An AI agent specializing in {", ".join(expertise)}.'
            }

    @property
    def name(self) -> str:
        return self._name

    @property
    def backstory(self) -> str:
        return self._backstory

    @property
    def bio(self) -> str:
        return self._bio

    @property
    def expertise(self) -> tuple:
        return self._expertise

    @property
    def decision_engine(self) -> "DecisionEngine":
        return self._decision_engine

    @property
    def personality(self) -> "Personality":
        return self._personality

    @abstractmethod
    def perceive(self, environment: "Environment") -> Dict[str, Any]:
        """
        Perceives the environment and returns a dictionary of perceptions.

        Args:
            environment: The environment to perceive.

        Returns:
            Dict[str, Any]: A dictionary of perceptions.
        """
        pass

    @abstractmethod
    def decide(self, perception: Dict[str, Any]) -> "Action":
        """
        Decides on an action based on the given perceptions.

        Args:
            perception: The perceptions to decide on.

        Returns:
            Action: The decided action.
        """
        pass

    @abstractmethod
    def act(self, action: "Action", environment: "Environment"):
        """
        Acts on the environment with the given action.

        Args:
            action: The action to act on.
            environment: The environment to act on.
        """
        pass

    @abstractmethod
    def update(self, feedback: Dict[str, Any]):
        """
        Updates the agent's state based on the given feedback.

        Args:
            feedback: The feedback to update the agent's state.
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        """
        Returns a string representation of the Agent.

        This method provides a meaningful string representation of the Agent's current state.

        Returns:
            str: A string representation of the Agent.
        """
        pass

    def listen(self, message: str) -> None:
        """
        Process incoming information or messages.

        Args:
            message (str): The incoming message or information.
        """
        # For now, we'll just print the message. In a more complex implementation,
        # this method might parse the message and update the agent's internal state.
        print(f"{self.name} is listening: {message}")

    def think(self, context: Dict[str, Any]) -> str:
        """
        Process information and generate thoughts or decisions.

        Args:
            context (Dict[str, Any]): The current context or situation.

        Returns:
            str: The agent's thought or decision.
        """
        prompt = f"""
        As {self.name}, an expert in {', '.join(self.expertise)}, think about the following context:
        {context}

        Given your personality and backstory:
        Personality: {self.personality}
        Backstory: {self.backstory}

        What are your thoughts or decisions? Respond in first person.
        """
        return self._language_model_handler.generate_text(prompt)

    def memorize(self, information: str) -> None:
        """
        Store important information in the agent's memory.

        Args:
            information (str): The information to be memorized.
        """
        self._memory.append(information)

    def remember(self, query: str) -> str:
        """
        Recall information from the agent's memory.

        Args:
            query (str): The query to search in the memory.

        Returns:
            str: The recalled information or a message if nothing is found.
        """
        memory_context = "\n".join(self._memory)
        prompt = f"""
        Given the following memory context:
        {memory_context}

        And the query: "{query}"

        What information can be recalled that's relevant to the query? If nothing is relevant, state that no relevant information was found.
        """
        return self._language_model_handler.generate_text(prompt)

    def talk(self, message: str) -> str:
        """
        Generate a response or statement.

        Args:
            message (str): The message or context to respond to.

        Returns:
            str: The agent's response.
        """
        prompt = f"""
        As {self.name}, with expertise in {', '.join(self.expertise)}, respond to the following message:
        "{message}"

        Consider your personality and backstory:
        Personality: {self.personality}
        Backstory: {self.backstory}

        Provide a response in first person, staying in character.
        """
        return self._language_model_handler.generate_text(prompt)


class Action(ABC):
    """
    Represents an action that can be taken by an agent.
    Defines how the action is executed in the environment.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Returns the name of the action.

        Returns:
            str: The name of the action.
        """
        pass

    @abstractmethod
    def execute(self, agent: Agent, environment: "Environment"):
        """
        Executes the action in the environment.

        Args:
            agent: The agent executing the action.
            environment: The environment to execute the action in.
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        """
        Returns a string representation of the Action.

        This method should be implemented by subclasses to provide a meaningful
        string representation of the Action's current state.

        Returns:
            str: A string representation of the Action.
        """
        pass




