import numpy as np
from typing import Tuple, List, Dict
from active_inference_forager.environments.chat_environment import ChatEnvironment
from active_inference_forager.utils.numpy_fields import NumpyArrayField
from pydantic import Field
from textblob import TextBlob

class PhilosophyDialogueEnvironment(ChatEnvironment):
    max_turns: int = Field(default=20)
    current_turn: int = Field(default=0)
    conversation_history: List[str] = Field(default_factory=list)
    user_understanding: Dict[str, float] = Field(default_factory=dict)
    user_interests: Dict[str, float] = Field(default_factory=dict)
    topic_engagement: Dict[str, float] = Field(default_factory=dict)
    current_topic: str = Field(default="")
    user_engagement: float = Field(default=0.5)

    def __init__(self, **data):
        super().__init__(**data)
        self.initialize_user_model()

    def initialize_user_model(self):
        topics = ["epistemology", "metaphysics", "ethics", "logic"]
        for topic in topics:
            self.user_understanding[topic] = np.random.uniform(0, 0.5)
            self.user_interests[topic] = np.random.uniform(0, 1)
            self.topic_engagement[topic] = np.random.uniform(0, 1)

    def reset(self) -> np.ndarray:
        self.current_turn = 0
        self.conversation_history = []
        self.initialize_user_model()
        self.current_topic = ""
        self.user_engagement = 0.5
        return self._get_state()

    def step(self, action: str, user_input: str) -> Tuple[np.ndarray, float, bool]:
        print(f"\nEnvironment step:")
        print(f"Action received: {action}")
        print(f"User input: {user_input}")

        self._simulate_user_response(action)
        self._process_user_input(user_input)
        self.current_turn += 1

        # Dynamic topic selection
        self._select_next_topic()

        reward = self._calculate_reward(action)
        done = self.current_turn >= self.max_turns or action == "end_conversation"

        new_state = self._get_state()
        print(f"New state: {new_state}")
        print(f"Reward: {reward}")
        print(f"Done: {done}")

        return new_state, reward, done

    def _get_state(self) -> np.ndarray:
        state = np.array(
            [
                self.current_turn / self.max_turns,
                1 - (self.current_turn / self.max_turns),  # time_remaining
                self.user_engagement,
                *[self.user_understanding[topic] for topic in sorted(self.user_understanding.keys())],
                *[self.user_interests[topic] for topic in sorted(self.user_interests.keys())],
                *[self.topic_engagement[topic] for topic in sorted(self.topic_engagement.keys())],
                sum(self.user_understanding.values()) / len(self.user_understanding),  # overall understanding
                sum(self.user_interests.values()) / len(self.user_interests),  # overall interest
            ]
        )
        print(f"State shape: {state.shape}")
        return state

    def _simulate_user_response(self, action: str):
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

    def _process_user_input(self, user_input: str):
        if not user_input.strip():
            return  # If the input is empty or only whitespace, do nothing

        blob = TextBlob(user_input)
        sentiment = blob.sentiment

        # Update user engagement based on sentiment
        self._update_user_engagement(sentiment.polarity * 0.1)

        # Update user interests based on keywords
        keywords = ["epistemology", "metaphysics", "ethics", "logic"]
        for keyword in keywords:
            if keyword in user_input.lower():
                self._update_user_interests(increase=0.05, topic=keyword)

        # Update user understanding based on the complexity of the response
        words = user_input.split()
        if words:
            avg_word_length = sum(len(word) for word in words) / len(words)
            if avg_word_length > 6:  # If the average word length is high, assume increased understanding
                self._update_user_understanding(increase=0.05)

    def _update_user_understanding(self, increase: float, topic: str = None):
        if topic is None:
            topic = self.current_topic
        if topic:
            self.user_understanding[topic] = min(1, self.user_understanding[topic] + increase)

    def _update_user_interests(self, increase: float, topic: str = None):
        if topic is None:
            topic = self.current_topic
        if topic:
            self.user_interests[topic] = min(1, self.user_interests[topic] + increase)

    def _update_user_engagement(self, change: float):
        self.user_engagement = max(0, min(1, self.user_engagement + change))
        if self.current_topic:
            self.topic_engagement[self.current_topic] = max(0, min(1, self.topic_engagement[self.current_topic] + change))

    def _calculate_reward(self, action: str) -> float:
        understanding_reward = sum(self.user_understanding.values()) / len(self.user_understanding)
        interest_reward = sum(self.user_interests.values()) / len(self.user_interests)
        engagement_reward = self.user_engagement

        return (understanding_reward + interest_reward + engagement_reward) / 3

    def set_current_topic(self, topic: str):
        if topic in self.user_understanding:
            self.current_topic = topic
        else:
            raise ValueError(f"Invalid topic: {topic}")

    def _select_next_topic(self):
        # Calculate a score for each topic based on interest and inverse of understanding
        topic_scores = {
            topic: self.user_interests[topic] * (1 - self.user_understanding[topic])
            for topic in self.user_understanding.keys()
        }
        
        # Select the topic with the highest score
        self.current_topic = max(topic_scores, key=topic_scores.get)
        print(f"Selected next topic: {self.current_topic}")
