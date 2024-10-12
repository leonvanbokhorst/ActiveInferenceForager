import numpy as np
from typing import Tuple, List, Dict
from active_inference_forager.environments.chat_environment import ChatEnvironment
from active_inference_forager.utils.numpy_fields import NumpyArrayField
from pydantic import Field


class PhilosophyDialogueEnvironment(ChatEnvironment):
    max_turns: int = Field(default=20)
    current_turn: int = Field(default=0)
    conversation_history: List[str] = Field(default_factory=list)
    user_understanding: Dict[str, float] = Field(default_factory=dict)
    user_interests: Dict[str, float] = Field(default_factory=dict)
    current_topic: str = Field(default="")
    user_engagement: float = Field(default=0.5)

    def __init__(self, **data):
        super().__init__(**data)
        self.initialize_user_model()

    def initialize_user_model(self):
        # Initialize user understanding and interests for each philosophical topic
        topics = ["epistemology", "metaphysics", "ethics", "logic"]
        for topic in topics:
            self.user_understanding[topic] = np.random.uniform(
                0, 0.5
            )  # Initial understanding is low to moderate
            self.user_interests[topic] = np.random.uniform(
                0, 1
            )  # Random initial interests

    def reset(self) -> np.ndarray:
        self.current_turn = 0
        self.conversation_history = []
        self.initialize_user_model()
        self.current_topic = ""
        self.user_engagement = 0.5
        return self._get_state()

    def step(self, action: str) -> Tuple[np.ndarray, float, bool]:
        print(f"\nEnvironment step:")
        print(f"Action received: {action}")

        # Simulate user response and update environment
        self._simulate_user_response(action)
        self.current_turn += 1

        # Calculate reward
        reward = self._calculate_reward(action)

        # Check if conversation is done
        done = self.current_turn >= self.max_turns or action == "end_conversation"

        new_state = self._get_state()
        print(f"New state: {new_state}")
        print(f"Reward: {reward}")
        print(f"Done: {done}")

        return new_state, reward, done

    def _calculate_reward(self, action: str) -> float:
        # ... existing reward calculation ...
        reward = (understanding_reward + interest_reward + engagement_reward) / 3
        print(f"Calculated reward: {reward}")
        return reward

    def _get_state(self) -> np.ndarray:
        # Construct state vector
        state = np.array(
            [
                self.current_turn / self.max_turns,
                self.user_engagement,
                *[
                    self.user_understanding[topic]
                    for topic in sorted(self.user_understanding.keys())
                ],
                *[
                    self.user_interests[topic]
                    for topic in sorted(self.user_interests.keys())
                ],
            ]
        )
        print(f"State shape: {state.shape}")  # Debug print
        return state

    def _simulate_user_response(self, action: str):
        # Simulate changes in user understanding, interests, and engagement based on the agent's action
        print(f"Simulating user response to action: {action}")

        if action == "explain_concept":
            self._update_user_understanding(increase=0.1)
            self._update_user_engagement(0.05)
        elif action == "ask_question":
            self._update_user_understanding(increase=0.05)
            self._update_user_engagement(0.1)
        elif action == "introduce_related_idea":
            self._update_user_interests(increase=0.1)
            self._update_user_engagement(0.05)
        elif action == "provide_example":
            self._update_user_understanding(increase=0.15)
            self._update_user_engagement(0.1)
        elif action == "suggest_thought_experiment":
            self._update_user_interests(increase=0.15)
            self._update_user_engagement(0.15)
        elif action == "acknowledge_limitation":
            self._update_user_engagement(-0.05)

        self.conversation_history.append(action)

    def _update_user_understanding(self, increase: float):
        if self.current_topic:
            self.user_understanding[self.current_topic] = min(
                1, self.user_understanding[self.current_topic] + increase
            )

    def _update_user_interests(self, increase: float):
        if self.current_topic:
            self.user_interests[self.current_topic] = min(
                1, self.user_interests[self.current_topic] + increase
            )

    def _update_user_engagement(self, change: float):
        self.user_engagement = max(0, min(1, self.user_engagement + change))

    def _calculate_reward(self, action: str) -> float:
        # Calculate reward based on changes in user understanding, interests, and engagement
        understanding_reward = sum(self.user_understanding.values()) / len(
            self.user_understanding
        )
        interest_reward = sum(self.user_interests.values()) / len(self.user_interests)
        engagement_reward = self.user_engagement

        return (understanding_reward + interest_reward + engagement_reward) / 3

    def set_current_topic(self, topic: str):
        if topic in self.user_understanding:
            self.current_topic = topic
        else:
            raise ValueError(f"Invalid topic: {topic}")

    # Additional helper methods can be added here as needed
