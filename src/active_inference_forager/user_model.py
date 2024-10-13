from typing import Dict

class UserModel:
    def __init__(self):
        self.preferences: Dict[str, float] = {}

    def update_preference(self, topic: str, score: float) -> None:
        self.preferences[topic] = score

    def get_preferences(self) -> Dict[str, float]:
        return self.preferences
