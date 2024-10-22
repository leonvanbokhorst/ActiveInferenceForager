from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np


class PersonalityTrait(ABC):
    """
    Characteristic that describes an agent's behavior.
    """

    @abstractmethod
    def __init__(self, name: str, value: Any):
        self.name = name
        self.value = value


    @abstractmethod
    def __str__(self) -> str:
        """
        Returns a string representation of the PersonalityTrait.

        This method should be implemented by subclasses to provide a meaningful
        string representation of the PersonalityTrait's current state.

        Returns:
            str: A string representation of the PersonalityTrait.
        """
        pass


class Personality(ABC):
    """
    Collection of personality traits that describe an individual's behavior.

    Args:
        ABC (_type_): _description_
    """

    @abstractmethod
    def __init__(self, traits: List[PersonalityTrait]):
        """
        Initialize the personality with a list of traits.

        Args:
            traits: The list of traits to initialize the personality with.
        """
        self.traits = traits

    @property
    def trait_vector(self) -> np.ndarray:
        """
        Vector representation of the personality traits.

        Returns:
            np.ndarray: A numpy array representing the personality traits.
        """
        pass

    @abstractmethod
    def similarity(self, other: "Personality") -> float:
        """
        Calculate the similarity between two personalities.

        Returns:
            float: A value between 0 (completely different) and 1 (identical).
        """
        pass


    @abstractmethod
    def __str__(self) -> str:
        """
        Returns a string representation of the Personality.

        This method should be implemented by subclasses to provide a meaningful
        string representation of the Personality's current state.

        Returns:
            str: A string representation of the Personality.
        """
        pass
