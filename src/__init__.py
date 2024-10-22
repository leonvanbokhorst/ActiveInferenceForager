# Add this line at the beginning of the file, after the other imports
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union
import numpy as np
# from .personality import Personality, PersonalityTrait

# Instead, you might want to import the mas_dynamics_simulation package
from . import mas_dynamics_simulation

# Add this import at the top of the file
import pytest
from mas_dynamics_simulation.personality_big_five import (
    BigFivePersonalityTrait,
    Openness,
    Conscientiousness,
    Extraversion,
    Agreeableness,
    Neuroticism,
    BigFivePersonality
)





