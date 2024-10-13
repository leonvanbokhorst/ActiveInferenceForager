import json
from typing import Dict, List


class UserModel:
    def __init__(self):
        self.preferences: Dict[str, float] = {}
        self.interaction_history: List[Dict] = []
        self.last_topic: str = "general"

    def update_preference(self, topic: str, score: float) -> None:
        self.preferences[topic] = max(
            0, min(1, score)
        )  # Ensure score is between 0 and 1

    def get_preferences(self) -> Dict[str, float]:
        return self.preferences

    def add_interaction(self, user_input: str, ai_response: str, topic: str) -> None:
        self.interaction_history.append(
            {"user_input": user_input, "ai_response": ai_response, "topic": topic}
        )
        self.last_topic = topic

    def get_interaction_history(self) -> List[Dict]:
        return self.interaction_history

    def get_last_topic(self) -> str:
        return self.last_topic

    def save_to_file(self, filename: str) -> None:
        with open(filename, "w") as f:
            json.dump(
                {
                    "preferences": self.preferences,
                    "interaction_history": self.interaction_history,
                    "last_topic": self.last_topic,
                },
                f,
            )

    def load_from_file(self, filename: str) -> None:
        try:
            with open(filename, "r") as f:
                data = json.load(f)
                self.preferences = data.get("preferences", {})
                self.interaction_history = data.get("interaction_history", [])
                self.last_topic = data.get("last_topic", "general")
        except FileNotFoundError:
            print(
                f"No existing user model found at {filename}. Starting with a new model."
            )
