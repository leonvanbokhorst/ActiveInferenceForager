import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, TYPE_CHECKING, List
import numpy as np
import json
import re
from mas_dynamics_simulation.language_model.language_model_handler import LanguageModelHandler
from mas_dynamics_simulation.decision_making import DecisionEngine
from mas_dynamics_simulation.personality import Personality

if TYPE_CHECKING:
    from .environment import Environment
    from .personality import Personality
    from .decision_making import DecisionEngine

# Set up logging
logger = logging.getLogger(__name__)

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
        details = self.generate_agent_details(expertise, decision_engine, language_model_handler, personality)
        self._name = details.get('name', 'Unknown Agent')
        self._backstory = details.get('backstory', 'No backstory available')
        self._bio = details.get('bio', 'No bio available')
        self._dark_secret = details.get('dark_secret', 'No dark secret available')
        self._hobbies = details.get('hobbies', '')
        self._interests = details.get('interests', '')
        self._goals = details.get('goals', '')
        self._fears = details.get('fears', '')
        self._strengths = details.get('strengths', '')
        self._weaknesses = details.get('weaknesses', '')
        self._quirks = details.get('quirks', '')
        self._public_behavior = details.get('public_behavior', '')
        self._private_behavior = details.get('private_behavior', '')
        self._health_issues = details.get('health_issues', '')  
        self._expertise = tuple(expertise)  # Make expertise immutable
        self._decision_engine = decision_engine
        self._personality = personality

        self._memory = []  # Initialize an empty memory

    @classmethod
    def generate_agent_details(cls, expertise: List[str], decision_engine: "DecisionEngine", language_model_handler: LanguageModelHandler, personality: Personality) -> Dict[str, str]:
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
        Given a researcher is a well-regarded expert in these fields: {', '.join(expertise)} 
        Furthermore, they have the following Big Five personality traits: {personality}, and very distinct character traits.
        
        1. Generate a name: a long, serious, realistic, cademic name in Honorific Style with using titles like Dr., Prof., Mr., Mrs. and middle names initials only for the researcher.
        2. Generate a backstory: from a third person perspective about the researcher's upbringing, education, career, culture, cities, countries, etc. text only.
        3. Generate a bio: from a third person perspective, and without explicitly naming personality traits, openly brag about the researcher's current role, achievements, superpowers, passions, quilty pleasures, and interests. text only.
        4. Generate a dark secret: awkward secrets, traumatic events, and/or strong biases for the researcher related to their background, upbringing, culture, etc. text only.
        5. Generate a hobbies of the researcher that are related to their personality, expertise, upbringing, and interests. Text only.    
        6. Generate a interests of the researcher that are related to their personality, expertise, upbringing, and hobbies. Text only.
        7. Generate a goals of the researcher that are related to their personality, expertise, upbringing, and interests. Text only.
        8. Generate a fears of the researcher that are related to their personality, expertise, upbringing, and interests. Text only.
        9. Generate a strengths of the researcher that are related to their personality, expertise, upbringing, and interests. Text only.
        10. Generate a weaknesses of the researcher that are related to their personality, expertise, upbringing, and interests. Text only.
        11. Generate a quirks of the researcher that are related to their personality, expertise, upbringing, and interests. Text only.
        12. Generate a typical public behavior of the researcher that is related to their personality, expertise, upbringing, and interests . Text only.
        13. Generate a typical private behavior of the researcher that is related to their personality, expertise, upbringing, and interests. Text only.
        14. Generate health issues of the researcher that are related to their personality, expertise, upbringing, and interests. Text only.
        
        Format the response ONLY as a JSON object with keys 'name', 'backstory', 'bio', 'dark_secret', 'hobbies', 'interests', 'goals', 'fears', 'strengths', 'weaknesses', 'quirks', 'public_behavior', 'private_behavior', 'health_issues'. No other text or markdown formatting.
        Example: {{"name": "Name text", "backstory": "Backstory text", "bio": "Bio text", "dark_secret": "Dark secret text", "hobbies": "Hobby 1, Hobby 2, Hobby 3", "interests": "Interest 1, Interest 2, Interest 3", "goals": "Goal 1, Goal 2, Goal 3", "fears": "Fear 1, Fear 2, Fear 3", "strengths": "Strength 1, Strength 2, Strength 3", "weaknesses": "Weakness 1, Weakness 2, Weakness 3", "quirks": "Quirk 1, Quirk 2, Quirk 3", "public_behavior": "Public behavior text", "private_behavior": "Private behavior text", "health_issues": "Health issue text"}}
        """
        logger.info(f"\nPrompt: {prompt}")

        max_attempts = 3
        for attempt in range(max_attempts):
            response = language_model_handler.generate_text(prompt)
            try:
                details = json.loads(response)
                
                # Check for special characters in the values
                for key, value in details.items():
                    if not isinstance(value, str):
                        raise ValueError(f"Value for {key} is not a string")
                    if re.search(r'[<>{}[\]]', value):
                        raise ValueError(f"Invalid characters (HTML tags or JSON brackets) in {key}")
                
                logger.info(f"Valid response generated on attempt {attempt + 1}")
                return details
            except json.JSONDecodeError as e:
                logger.error(f"\nAttempt {attempt + 1} failed. Invalid JSON: {e}")
            except ValueError as e:
                logger.error(f"\nAttempt {attempt + 1} failed. {e}")
            
            logger.debug(f"Response: {response}")
            
            if attempt == max_attempts - 1:
                logger.warning("Max attempts reached. Using default values.")
                return {
                    "name": "Default Agent",
                    "backstory": "No backstory available",
                    "bio": "No bio available",
                    "dark_secret": "No dark secret available",
                    "hobbies": "",
                    "interests": "",
                    "goals": "",
                    "fears": "",
                    "strengths": "",
                    "weaknesses": "",
                    "quirks": ""
                }
        
        # This line should never be reached, but it's here for completeness
        return {
            "name": "Default Agent",
            "backstory": "No backstory available",
            "bio": "No bio available",
            "dark_secret": "No dark secret available",
            "hobbies": "",
            "interests": "",
            "goals": "",
            "fears": "",
            "strengths": "",
            "weaknesses": "",
            "quirks": ""
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
    
    @property
    def dark_secret(self) -> str:
        return self._dark_secret
    
    @property
    def hobbies(self) -> str:
        return self._hobbies
    
    @property
    def interests(self) -> str:
        return self._interests
    
    @property
    def goals(self) -> str:
        return self._goals
    
    @property
    def fears(self) -> str:
        return self._fears
    
    @property
    def strengths(self) -> str:
        return self._strengths
    
    @property
    def weaknesses(self) -> str:
        return self._weaknesses
    
    @property
    def quirks(self) -> str:
        return self._quirks
    
    @property
    def public_behavior(self) -> str:
        return self._public_behavior
    
    @property
    def private_behavior(self) -> str:
        return self._private_behavior
    
    @property
    def health_issues(self) -> str:
        return self._health_issues
    
    @property
    def memory(self) -> List[str]:
        return self._memory

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
        logger.info(f"{self.name} is listening: {message}")

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
        logger.debug(f"Thinking prompt: {prompt}")
        return self._language_model_handler.generate_text(prompt)

    def memorize(self, information: str) -> None:
        """
        Store important information in the agent's memory.

        Args:
            information (str): The information to be memorized.
        """
        self._memory.append(information)
        logger.debug(f"{self.name} memorized: {information}")

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
        logger.debug(f"Remember prompt: {prompt}")
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
        logger.debug(f"Talk prompt: {prompt}")
        response = self._language_model_handler.generate_text(prompt)
        logger.info(f"{self.name} says: {response}")
        return response


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





