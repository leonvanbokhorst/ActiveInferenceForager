import numpy as np

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union


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
    def __init__(self, traits: Dict[str, Union[float, PersonalityTrait]] = None):
        """
        Initialize the personality with a list of traits.

        Args:
            traits: The list of traits to initialize the personality with.
        """

    @property
    def traits(self) -> Dict[str, PersonalityTrait]:
        """
        Dictionary representation of the personality traits.

        Returns:
            Dict[str, PersonalityTrait]: A dictionary representing the personality traits.
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

    def __getitem__(self, trait_name: str) -> PersonalityTrait:
        return self.traits[trait_name.lower()]

    def __setitem__(self, trait_name: str, value: Union[float, PersonalityTrait]):
        if trait_name.lower() in self.traits:
            if isinstance(value, PersonalityTrait):
                self.traits[trait_name.lower()] = value
            elif isinstance(value, (int, float)):
                self.traits[trait_name.lower()].value = value
            else:
                raise ValueError(f"Invalid trait value for {trait_name}")
        else:
            raise ValueError(f"Unknown trait: {trait_name}")


class BigFivePersonalityTrait(PersonalityTrait):
    """
    A personality trait based on the Big Five model.
    """

    def __init__(self, name: str, value: float = 0.5, explanation: str = None):
        super().__init__(name, value)
        self.explanation = explanation
        self.value = self._validate_value(value)

    def _validate_value(self, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Value must be between 0.0 and 1.0, got {value}")
        return value

    @property
    def value(self) -> float:
        return self._value

    @value.setter
    def value(self, new_value: float):
        self._value = self._validate_value(new_value)

    def __str__(self) -> str:
        return f"{self.name}: {self.value:.2f} - {self.explanation}"

    def __eq__(self, other):
        if not isinstance(other, BigFivePersonalityTrait):
            return NotImplemented
        return (
            self.name == other.name
            and self.value == other.value
            and self.explanation == other.explanation
        )

    def __hash__(self):
        return hash((self.name, self.value, self.explanation))


class Openness(BigFivePersonalityTrait):
    def __init__(self):
        explanation = "Openness to experience. High scorers tend to be creative, curious, and adventurous."
        super().__init__("Openness", explanation=explanation)


class Conscientiousness(BigFivePersonalityTrait):
    def __init__(self):
        explanation = "Tendency to be organized, responsible, and hardworking."
        super().__init__("Conscientiousness", explanation=explanation)


class Extraversion(BigFivePersonalityTrait):
    def __init__(self):
        explanation = "Tendency to seek stimulation in the company of others, to be outgoing and energetic."
        super().__init__("Extraversion", explanation=explanation)


class Agreeableness(BigFivePersonalityTrait):
    def __init__(self):
        explanation = (
            "Tendency to be compassionate, cooperative, and trusting towards others."
        )
        super().__init__("Agreeableness", explanation=explanation)


class Neuroticism(BigFivePersonalityTrait):
    def __init__(self):
        explanation = "Tendency to experience negative emotions easily, such as anxiety, anger, or depression."
        super().__init__("Neuroticism", explanation=explanation)


class BigFivePersonality(Personality):
    """
    Implementation of the Big Five personality model.
    Each trait is scored on a scale from 0.0 to 1.0 (equivalent to 0-100%).
    """

    STD_DEV_NARROW = 0.12
    STD_DEV_NORMAL = 0.15

    LEVEL_DESCRIPTIONS = {
        (0.0, 0.2): "Very Low",
        (0.2, 0.4): "Low",
        (0.4, 0.6): "Moderate",
        (0.6, 0.8): "High",
        (0.8, 1.0): "Very High",
    }

    def __init__(self, traits: Dict[str, Union[float, PersonalityTrait]] = None):
        self._traits = {
            "openness": Openness(),
            "conscientiousness": Conscientiousness(),
            "extraversion": Extraversion(),
            "agreeableness": Agreeableness(),
            "neuroticism": Neuroticism(),
        }

        if traits:
            for trait_name, value in traits.items():
                self[trait_name.lower()] = value

    @property
    def traits(self) -> Dict[str, BigFivePersonalityTrait]:
        return self._traits

    def similarity(self, other: "BigFivePersonality") -> float:
        if not isinstance(other, BigFivePersonality):
            raise ValueError("Can only compare with another BigFivePersonality")

        vec1 = np.fromiter((trait.value for trait in self.traits.values()), dtype=float)
        vec2 = np.fromiter(
            (trait.value for trait in other.traits.values()), dtype=float
        )

        distance = np.linalg.norm(vec1 - vec2)
        max_distance = np.sqrt(5)
        similarity = 1 - (distance / max_distance)

        return similarity

    def compare(self, other: "BigFivePersonality", other_name: str) -> str:
        if not isinstance(other, BigFivePersonality):
            raise ValueError("Can only compare with another BigFivePersonality")

        differences = []
        trait_adjectives = {
            "openness": "open to experience",
            "conscientiousness": "conscientious",
            "extraversion": "extraverted",
            "agreeableness": "agreeable",
            "neuroticism": "emotionally stable",
        }

        THRESHOLD = 0.1

        for trait_name, trait in self.traits.items():
            diff = trait.value - other.traits[trait_name].value
            if trait_name == "neuroticism":
                diff = -diff
            if abs(diff) >= THRESHOLD:
                differences.append((trait_adjectives[trait_name], diff))

        if not differences:
            return f"You have a very similar personality to {other_name}."

        differences.sort(key=lambda x: abs(x[1]), reverse=True)

        more_traits = [trait for trait, diff in differences if diff > 0]
        less_traits = [trait for trait, diff in differences if diff < 0]

        parts = []

        for traits, prefix in [(more_traits, "more"), (less_traits, "less")]:
            if traits:
                if len(traits) == 1:
                    parts.append(f"{prefix} {traits[0]}")
                elif len(traits) == 2:
                    parts.append(f"{prefix} {traits[0]} and {traits[1]}")
                else:
                    traits_str = ", ".join(traits[:-1]) + f", and {traits[-1]}"
                    parts.append(f"{prefix} {traits_str}")

        return f"You are {' but '.join(parts)} than {other_name}."

    def __str__(self) -> str:
        return "\n".join(
            f"{trait.name}: {trait.value*100:.1f}% - {self._get_level_description(trait.value)}"
            for trait in self.traits.values()
        )

    @classmethod
    def _get_level_description(cls, value: float) -> str:
        LEVEL_DESCRIPTIONS = {
            (0.0, 0.2): "Very Low",
            (0.2, 0.4): "Low",
            (0.4, 0.6): "Moderate",
            (0.6, 0.8): "High",
            (0.8, 1.0): "Very High",
        }
        for (lower, upper), description in LEVEL_DESCRIPTIONS.items():
            if lower <= value < upper:
                return description
        return LEVEL_DESCRIPTIONS[(0.8, 1.0)]  # Default to "Very High" for edge case

    @staticmethod
    def _generate_realistic_value(
        mean: float, std_dev: float, volatility: float = 0.1
    ) -> float:
        """
        Generate a realistic personality trait value using a normal distribution,
        with added random fluctuation.

        Args:
            mean: The center point for the trait value
            std_dev: The standard deviation for the normal distribution
            volatility: How much the mean can shift randomly
        """
        adjusted_mean = mean + np.random.uniform(-volatility, volatility)
        adjusted_mean = max(0.1, min(0.9, adjusted_mean))

        if np.random.random() < 0.15:  # 15% chance of more extreme value
            std_dev *= 1.5

        while True:
            value = np.random.normal(adjusted_mean, std_dev)
            if 0.0 <= value <= 1.0:
                return value

    @classmethod
    def random(
        cls, variation: str = "balanced", volatility: float = 0.1
    ) -> "BigFivePersonality":
        """
        Creates a personality with realistic but varying trait values.

        Args:
            variation (str): The type of personality to generate:
                - "balanced": Traits centered around the middle with natural variation
                - "gentle": Generally higher agreeableness and conscientiousness
                - "bold": Generally higher extraversion and openness
                - "analytical": Generally higher conscientiousness and openness
                - "random": Completely randomized but still realistic traits
            volatility (float): How much the traits can deviate from their typical values (0.0-1.0)

        Returns:
            BigFivePersonality: A new personality instance
        """
        trait_params = {
            "openness": (0.5, cls.STD_DEV_NORMAL),
            "conscientiousness": (0.5, cls.STD_DEV_NORMAL),
            "extraversion": (0.5, cls.STD_DEV_NORMAL),
            "agreeableness": (0.5, cls.STD_DEV_NORMAL),
            "neuroticism": (0.5, cls.STD_DEV_NORMAL),
        }

        if variation == "random":
            trait_params = {
                trait: (np.random.uniform(0.3, 0.7), 0.2)
                for trait in trait_params.keys()
            }
        elif variation == "gentle":
            trait_params.update(
                {
                    "agreeableness": (0.7, cls.STD_DEV_NARROW),
                    "conscientiousness": (0.65, cls.STD_DEV_NARROW),
                    "neuroticism": (0.35, cls.STD_DEV_NARROW),
                    "openness": (0.5, 0.2),
                    "extraversion": (0.5, 0.2),
                }
            )
        elif variation == "bold":
            trait_params.update(
                {
                    "extraversion": (0.65, cls.STD_DEV_NARROW),
                    "openness": (0.6, cls.STD_DEV_NARROW),
                    "neuroticism": (0.45, cls.STD_DEV_NORMAL),
                    "conscientiousness": (0.5, 0.2),
                    "agreeableness": (0.5, 0.2),
                }
            )
        elif variation == "analytical":
            trait_params.update(
                {
                    "conscientiousness": (0.65, cls.STD_DEV_NARROW),
                    "openness": (0.6, cls.STD_DEV_NARROW),
                    "extraversion": (0.4, cls.STD_DEV_NARROW),
                    "agreeableness": (0.5, cls.STD_DEV_NORMAL),
                    "neuroticism": (0.5, cls.STD_DEV_NORMAL),
                }
            )

        random_traits = {}
        for trait, (mean, std_dev) in trait_params.items():
            value = cls._generate_realistic_value(mean, std_dev, volatility)
            random_traits[trait] = value

        if np.random.random() < 0.6:  # 60% chance of correlated traits
            if random_traits["conscientiousness"] > 0.6:
                random_traits["neuroticism"] = min(
                    random_traits["neuroticism"],
                    cls._generate_realistic_value(0.4, 0.15, volatility),
                )
            if random_traits["extraversion"] > 0.6:
                random_traits["openness"] = max(
                    random_traits["openness"],
                    cls._generate_realistic_value(0.6, 0.15, volatility),
                )
            if random_traits["neuroticism"] > 0.7:
                random_traits["extraversion"] = min(
                    random_traits["extraversion"],
                    cls._generate_realistic_value(0.4, 0.15, volatility),
                )

        return cls(random_traits)

    def similarity_description(
        self, other: "BigFivePersonality", other_name: str
    ) -> str:
        similarity = self.similarity(other)

        if similarity >= 0.95:
            description = "very similar"
        elif similarity >= 0.85:
            description = "similar"
        elif similarity >= 0.75:
            description = "somewhat similar"
        elif similarity >= 0.65:
            description = "somewhat different"
        elif similarity >= 0.50:
            description = "quite different"
        else:
            description = "very different"

        return f"You and {other_name} have {description} personalities."


class DefaultPersonality(Personality):
    def __init__(self):
        # Initialize with default values
        pass

    def __str__(self):
        return "Default Personality"

    def similarity(self, other: "Personality") -> float:
        # Implement a basic similarity measure
        return 0.5 if isinstance(other, DefaultPersonality) else 0.0
