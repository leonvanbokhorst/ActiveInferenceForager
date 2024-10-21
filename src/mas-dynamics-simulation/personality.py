from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np


class PersonalityTrait(ABC):
    """
    Characteristic that describes an individual's behavior.

    Args:
        ABC (_type_): _description_
    """

    @abstractmethod
    def __init__(self, name: str, value: Any):
        self.name = name
        self.value = value

    @abstractmethod
    def to_numeric(self) -> float:
        """Convert the trait value to a numeric representation."""
        pass

    @abstractmethod
    def from_numeric(self, value: float):
        """Set the trait value from a numeric representation."""
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass


class Personality(ABC):
    """
    Collection of personality traits that describe an individual's behavior.

    Args:
        ABC (_type_): _description_
    """

    @abstractmethod
    def __init__(self, traits: List[PersonalityTrait]):
        self.traits = traits

    @property
    def trait_vector(self) -> np.ndarray:
        """Return a numpy array representing the personality traits."""
        return np.array([trait.to_numeric() for trait in self.traits])

    def similarity(self, other: "Personality") -> float:
        """
        Calculate the similarity between two personalities.
        Returns a value between 0 (completely different) and 1 (identical).
        """
        vec1 = self.trait_vector
        vec2 = other.trait_vector
        return 1 - (np.linalg.norm(vec1 - vec2) / (np.sqrt(2) * len(self.traits)))

    @abstractmethod
    def influence_decision(self, decision_context: Dict[str, Any]) -> Dict[str, float]:
        """
        Determine how the personality influences a decision.
        Returns a dictionary of influence factors for different aspects of the decision.
        """
        pass

    def __str__(self) -> str:
        return f"Personality({', '.join(str(trait) for trait in self.traits)})"
