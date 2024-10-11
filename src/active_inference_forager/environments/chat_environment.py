import numpy as np
from typing import Tuple, List
from active_inference_forager.environments.base_environment import BaseEnvironment
from active_inference_forager.utils.numpy_fields import NumpyArrayField
from pydantic import Field

class ChatEnvironment(BaseEnvironment):
    max_turns: int = Field(default=10)
    current_turn: int = Field(default=0)
    conversation_history: List[str] = Field(default_factory=list)
    user_satisfaction: float = Field(default=0.0)
    task_completion: float = Field(default=0.0)

    def __init__(self, **data):
        state_dim = 5  # conversation length, user satisfaction, task completion, last action, context relevance
        action_dim = 6  # number of possible actions
        super().__init__(state_dim=state_dim, action_dim=action_dim, **data)
        self.reset()

    @property
    def action_space(self) -> np.ndarray:
        return np.array([
            "ask_question",
            "provide_information",
            "clarify",
            "suggest_action",
            "express_empathy",
            "end_conversation"
        ])

    def reset(self) -> np.ndarray:
        self.current_turn = 0
        self.conversation_history = []
        self.user_satisfaction = 0.5  # Start with neutral satisfaction
        self.task_completion = 0.0
        self.state = self._get_state()
        return self.state

    def step(self, action: str) -> Tuple[np.ndarray, float, bool]:
        if action not in self.action_space:
            raise ValueError(f"Invalid action: {action}")

        # Simulate user response and update environment
        self._simulate_user_response(action)
        self.current_turn += 1

        # Calculate reward
        reward = self._calculate_reward(action)

        # Check if conversation is done
        done = self.current_turn >= self.max_turns or action == "end_conversation"

        self.state = self._get_state()
        return self.state, reward, done

    def _get_state(self) -> np.ndarray:
        last_action = self.conversation_history[-1] if self.conversation_history else None
        context_relevance = np.random.uniform(0, 1)  # Simulated context relevance
        
        # Handle the case when there's no last action
        if last_action is None:
            last_action_index = 0  # Use 0 as default when there's no last action
        else:
            last_action_index = np.where(self.action_space == last_action)[0][0]
        
        return np.array([
            self.current_turn / self.max_turns,
            self.user_satisfaction,
            self.task_completion,
            last_action_index / len(self.action_space),
            context_relevance
        ])

    def _simulate_user_response(self, action: str):
        # Simulate changes in user satisfaction and task completion based on the agent's action
        if action == "ask_question":
            self.user_satisfaction += np.random.uniform(-0.1, 0.2)
            self.task_completion += np.random.uniform(0, 0.1)
        elif action == "provide_information":
            self.user_satisfaction += np.random.uniform(-0.1, 0.3)
            self.task_completion += np.random.uniform(0.1, 0.3)
        elif action == "clarify":
            self.user_satisfaction += np.random.uniform(0, 0.2)
            self.task_completion += np.random.uniform(0, 0.1)
        elif action == "suggest_action":
            self.user_satisfaction += np.random.uniform(-0.2, 0.3)
            self.task_completion += np.random.uniform(0.1, 0.4)
        elif action == "express_empathy":
            self.user_satisfaction += np.random.uniform(0.1, 0.3)
        elif action == "end_conversation":
            self.user_satisfaction += np.random.uniform(-0.3, 0.1)

        self.user_satisfaction = np.clip(self.user_satisfaction, 0, 1)
        self.task_completion = np.clip(self.task_completion, 0, 1)
        self.conversation_history.append(action)

    def _calculate_reward(self, action: str) -> float:
        # Calculate reward based on user satisfaction, task completion, and conversation coherence
        satisfaction_reward = self.user_satisfaction - 0.5  # Reward for satisfaction above neutral
        completion_reward = self.task_completion
        coherence_reward = self._calculate_coherence_reward(action)
        
        return satisfaction_reward + completion_reward + coherence_reward

    def _calculate_coherence_reward(self, action: str) -> float:
        if len(self.conversation_history) < 2:
            return 0
        
        last_action = self.conversation_history[-2]
        coherence_matrix = {
            "ask_question": {"clarify": 0.1, "provide_information": -0.1},
            "provide_information": {"ask_question": 0.1, "suggest_action": 0.2},
            "clarify": {"provide_information": 0.2, "ask_question": -0.1},
            "suggest_action": {"end_conversation": 0.1, "ask_question": -0.1},
            "express_empathy": {"ask_question": 0.1, "provide_information": 0.1},
        }
        
        return coherence_matrix.get(last_action, {}).get(action, 0)
