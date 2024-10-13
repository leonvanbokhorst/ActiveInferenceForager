import json
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class UserModel:
    def __init__(self):
        self.preferences: Dict[str, float] = {}

    def update_preference(self, topic: str, score: float) -> None:
        self.preferences[topic] = score

    def get_preferences(self) -> Dict[str, float]:
        return self.preferences

    def save_to_file(self, filename: str) -> None:
        with open(filename, "w") as f:
            json.dump(self.preferences, f)

    def load_from_file(self, filename: str) -> None:
        try:
            with open(filename, "r") as f:
                self.preferences = json.load(f)
        except FileNotFoundError:
            logger.warning(
                f"User model file {filename} not found. Starting with empty preferences."
            )
