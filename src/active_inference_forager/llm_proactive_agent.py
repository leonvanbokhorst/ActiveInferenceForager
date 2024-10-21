from typing import List, Dict, Any
from active_inference_forager.managers.rapport_builder import RapportBuilder
from active_inference_forager.managers.goal_seeker import GoalSeeker
from active_inference_forager.providers.llm_provider import LLMProvider
from active_inference_forager.models.llm_generative_model import LLMGenerativeModel

import logging
from logging.handlers import RotatingFileHandler

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create handlers
file_handler = RotatingFileHandler(
    "logs/proactive_agent.log", maxBytes=1000000, backupCount=3
)
console_handler = logging.StreamHandler()

# Create formatters and add it to handlers
log_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(log_format)
console_handler.setFormatter(log_format)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)


class LLMProactiveAgent:
    def __init__(
        self,
        goal_seeker: GoalSeeker,
        proactive_threshold: float = 5.0,
    ):
        self.goal_seeker = goal_seeker
        self.conversation_history: List[Dict[str, str]] = []
        self.current_state: Dict[str, Any] = {}
        self.current_free_energy: float = float("inf")
        self.beliefs: Dict[str, Any] = {}
        self.proactive_threshold: float = proactive_threshold
        self.proactive_action_history: List[Dict[str, Any]] = []
        self.user_boundaries = set()
        self.ethical_violation_count = 0
        self.max_ethical_violations = 3

    def check_ethical_constraints(self, proposed_action):
        if any(boundary in proposed_action for boundary in self.user_boundaries):
            self.ethical_violation_count += 1
            if self.ethical_violation_count >= self.max_ethical_violations:
                return "TERMINATE_CONVERSATION"
            return False
        return True

    def update_user_boundaries(self, user_input):
        negative_phrases = ["leave me alone", "not interested", "stop", "go away"]
        for phrase in negative_phrases:
            if phrase in user_input.lower():
                self.user_boundaries.add(phrase)

    def process_user_input(self, user_input: str) -> str:
        self.update_user_boundaries(user_input)
        observations = self.goal_seeker.extract_features(user_input)
        self.goal_seeker.rapid_goal_adjustment(observations["sentiment"], user_input)
        self.goal_seeker.inference_engine.rapid_belief_update(
            user_input, observations["sentiment"]
        )

        response = self.goal_seeker.process_input(user_input)
        if not self.check_ethical_constraints(response):
            return "I apologize, but I think it's best if we end our conversation here. Thank you for your time."

        self._update_conversation_history(user_input, response)
        return response

    def _update_beliefs(self, observations: Dict[str, Any]):
        updated_beliefs = self.goal_seeker.inference_engine.infer(observations)
        self.beliefs.update(updated_beliefs)
        logger.debug(f"Updated beliefs: {self.beliefs}")

    def _update_conversation_history(self, user_input: str, agent_response: str):
        self.conversation_history.append({"user": user_input, "agent": agent_response})
        self.conversation_history = self.conversation_history[-10:]
        logger.debug(
            f"Updated conversation history. Current length: {len(self.conversation_history)}"
        )

    def run(self):
        while True:
            user_input = input("User: ")
            if user_input.lower() == "exit":
                logger.info("Conversation ended by user")
                logger.info("Agent: Goodbye!")
                break
            response = self.process_user_input(user_input)
            logger.info(f"Agent: {response}")

    def set_initial_goal(self, goal: str):
        self.goal_seeker.set_goal(goal)
        self.current_free_energy = self.goal_seeker.get_current_free_energy()
        logger.info(f"Initial free energy: {self.current_free_energy}")

    def get_current_goal(self) -> str:
        return self.goal_seeker.current_goal

    def get_goal_hierarchy(self) -> List[str]:
        return self.goal_seeker.goal_hierarchy

    def get_current_state(self) -> Dict[str, Any]:
        return self.current_state

    def get_current_free_energy(self) -> float:
        return self.current_free_energy

    def get_beliefs(self) -> Dict[str, Any]:
        return self.beliefs

    def visualize_goal_hierarchy(self):
        for i, goal in enumerate(self.get_goal_hierarchy()):
            logger.info(f"{i+1}. {goal}")

    def handle_proactive_behavior(self):
        self._adjust_proactive_threshold()
        current_goal = self.get_current_goal()
        goal_hierarchy = self.get_goal_hierarchy()
        current_state = self.get_current_state()
        beliefs = self.get_beliefs()
        recent_energy_changes = self.goal_seeker.get_recent_energy_changes()
        # convert np floats to floats
        recent_energy_changes = [float(x) for x in recent_energy_changes]
        proactive_action = self._generate_proactive_action(
            current_goal, goal_hierarchy, current_state, beliefs, recent_energy_changes
        )
        if proactive_action:
            self._execute_proactive_action(proactive_action)
        self._evaluate_proactive_action(proactive_action)

    def _generate_proactive_action(
        self,
        current_goal: str,
        goal_hierarchy: List[str],
        current_state: Dict[str, Any],
        beliefs: Dict[str, Any],
        recent_energy_changes: List[float],
    ) -> Dict[str, Any]:
        logger.info("Generating proactive action")
        action_effectiveness = self._analyze_past_actions()
        conversation_context = self._summarize_conversation_history()
        user_sentiment = self._analyze_user_sentiment()
        prompt = f"""
        Based on the following information, generate a proactive action for the agent to take:
        
        Current goal: {current_goal}
        Goal hierarchy: {goal_hierarchy}
        Current state: {current_state}
        Beliefs: {beliefs}
        Current free energy: {self.current_free_energy}
        Proactive threshold: {self.proactive_threshold}
        Past action effectiveness: {action_effectiveness}
        Conversation context: {conversation_context}
        User sentiment: {user_sentiment}
        Recent energy changes: {recent_energy_changes}
        
        Consider the following types of proactive actions:
        1. Adjust the current goal
        2. Propose a new sub-goal
        3. Update beliefs based on new information
        4. Suggest a course of action to the user
        5. Ask a clarifying question to gather more information
        6. Provide additional information or context to the user
        7. Summarize the conversation or progress so far
        
        Use the past action effectiveness, conversation context, user sentiment, and recent energy changes to inform your decision. 
        Prefer actions that have been more effective in the past and align with the user's current sentiment.
        If recent energy changes show consistent improvement, consider more ambitious actions.
        If recent energy changes show stagnation or increase, consider more conservative or information-gathering actions.
        
        Provide the proactive action in the following format:
        {{"type": "action_type", "description": "detailed description of the action", "reason": "reason for taking this action"}}
        """
        logger.debug(f"Proactive action prompt: {prompt}")
        response = self.goal_seeker.llm_provider.generate_response(prompt)
        proactive_action = eval(response)
        return proactive_action

    def _execute_proactive_action(self, action: Dict[str, Any]):
        logger.info(f"Executing proactive action: {action}")
        action_type = action["type"]
        description = action["description"]
        if action_type == "adjust_goal":
            self.set_initial_goal(description)
        elif action_type == "propose_subgoal":
            self.goal_seeker.goal_hierarchy.insert(1, description)
        elif action_type == "update_beliefs":
            self.beliefs.update(eval(description))
        elif action_type == "suggest_action":
            logger.info(f"Suggesting action to user: {description}")
        elif action_type == "ask_question":
            logger.info(f"Asking clarifying question: {description}")
        elif action_type == "provide_information":
            logger.info(f"Providing additional information: {description}")
        elif action_type == "summarize":
            logger.info(f"Summarizing conversation: {description}")
        self.proactive_action_history.append(action)

    def _evaluate_proactive_action(self, action: Dict[str, Any]):
        logger.info("Evaluating proactive action")
        if not action:
            return
        new_free_energy = self.goal_seeker.get_current_free_energy()
        energy_change = self.current_free_energy - new_free_energy
        logger.info(f"Free energy change after proactive action: {energy_change}")
        self.current_free_energy = new_free_energy
        action["energy_change"] = energy_change
        action["effectiveness"] = (
            "high" if energy_change > 1 else "medium" if energy_change > 0 else "low"
        )
        logger.info(f"Proactive action effectiveness: {action['effectiveness']}")

    def _analyze_past_actions(self) -> Dict[str, float]:
        logger.info("Analyzing past actions")
        action_effectiveness = {}
        for action in self.proactive_action_history:
            action_type = action["type"]
            effectiveness = action["effectiveness"]
            if action_type not in action_effectiveness:
                action_effectiveness[action_type] = []
            action_effectiveness[action_type].append(
                1.0
                if effectiveness == "high"
                else 0.5 if effectiveness == "medium" else 0.0
            )
        for action_type, scores in action_effectiveness.items():
            action_effectiveness[action_type] = sum(scores) / len(scores)
        logger.info(f"Action effectiveness analysis: {action_effectiveness}")
        return action_effectiveness

    def _adjust_proactive_threshold(self):
        logger.info("Adjusting proactive threshold")
        recent_energy_changes = self.goal_seeker.get_recent_energy_changes()
        if not recent_energy_changes:
            return
        avg_energy_change = sum(recent_energy_changes) / len(recent_energy_changes)
        if avg_energy_change > 0:
            self.proactive_threshold *= (
                0.9  # Decrease threshold if we're making progress
            )
        else:
            self.proactive_threshold *= (
                1.1  # Increase threshold if we're not making progress
            )
        self.proactive_threshold = max(
            1.0, min(10.0, self.proactive_threshold)
        )  # Keep threshold between 1 and 10
        logger.info(f"Adjusted proactive threshold: {self.proactive_threshold}")

    def _summarize_conversation_history(self) -> str:
        logger.info("Summarizing conversation history")
        if not self.conversation_history:
            return "No conversation history available."

        prompt = f"""
        Summarize the following conversation history in a concise manner, highlighting key points and themes:
        
        {self._extract_conversation_context()}
        
        Provide a summary that captures the main topics discussed, any decisions made, and the overall direction of the conversation.
        """

        summary = self.goal_seeker.llm_provider.generate_response(prompt)
        logger.info(f"Conversation summary: {summary}")
        return summary

    def _extract_conversation_context(self) -> str:
        logger.info("Extracting conversation context")
        recent_messages = self.conversation_history[-5:]  # Get the last 5 messages
        context = "\n".join(
            [f"User: {msg['user']}\nAgent: {msg['agent']}" for msg in recent_messages]
        )
        return context

    def _analyze_user_sentiment(self) -> str:
        logger.info("Analyzing user sentiment")
        if not self.conversation_history:
            return "neutral"
        last_user_message = self.conversation_history[-1]["user"]
        prompt = f"""
        Analyze the sentiment of the following user message:
        "{last_user_message}"
        Provide the sentiment as one of: positive, neutral, or negative.
        """
        sentiment = (
            self.goal_seeker.llm_provider.generate_response(prompt).strip().lower()
        )
        logger.info(f"Detected user sentiment: {sentiment}")
        return sentiment
