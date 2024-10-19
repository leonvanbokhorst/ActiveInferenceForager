from typing import List, Dict, Any
from active_inference_forager.managers.rapport_builder import RapportBuilder
from active_inference_forager.managers.goal_seeker import GoalSeeker
from active_inference_forager.providers.llm_provider import LLMProvider
from active_inference_forager.models.llm_generative_model import LLMGenerativeModel

import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LLMProactiveAgent:
    def __init__(self, rapport_builder: RapportBuilder, goal_seeker: GoalSeeker, proactive_threshold: float = 5.0):
        self.rapport_builder = rapport_builder
        self.goal_seeker = goal_seeker
        self.conversation_history: List[Dict[str, str]] = []
        self.current_state: Dict[str, Any] = {}
        self.current_free_energy: float = float("inf")
        self.beliefs: Dict[str, Any] = {}
        self.proactive_threshold: float = proactive_threshold
        logger.info("LLMProactiveAgent initialized")

    def process_user_input(self, user_input: str) -> str:
        logger.info(f"Processing user input: {user_input}")

        # Extract features from user input
        observations = self.goal_seeker.extract_features(user_input)
        logger.debug(f"Extracted features: {observations}")

        # Update current state
        self.current_state.update(observations)

        # Minimize free energy and potentially update goals
        new_free_energy = self.goal_seeker.minimize_free_energy()

        # Check if we've reduced free energy
        if new_free_energy < self.current_free_energy:
            logger.info(
                f"Free energy reduced from {self.current_free_energy} to {new_free_energy}"
            )
            self.current_free_energy = new_free_energy
        else:
            logger.info(f"Free energy remained at {self.current_free_energy}")

        # Update beliefs based on new observations
        self._update_beliefs(observations)

        # Generate response
        response = self.goal_seeker.process_input(user_input)
        logger.debug(f"Goal-oriented response: {response}")

        # Build rapport
        rapport_response = self.rapport_builder.process_input(user_input)
        logger.debug(f"Rapport-building response: {rapport_response}")

        # Combine goal-oriented response with rapport-building elements
        final_response = self._combine_responses(response, rapport_response)

        # Update conversation history
        self._update_conversation_history(user_input, final_response)

        logger.info(f"Final response: {final_response}")
        return final_response

    def _update_beliefs(self, observations: Dict[str, Any]):
        logger.info("Updating beliefs based on new observations")
        updated_beliefs = self.goal_seeker.inference_engine.infer(observations)
        self.beliefs.update(updated_beliefs)
        logger.debug(f"Updated beliefs: {self.beliefs}")

    def _combine_responses(self, goal_response: str, rapport_response: str) -> str:
        logger.info("Combining goal-oriented and rapport-building responses")
        prompt = f"""
        Combine the following two responses into a single, coherent response:
        
        Goal-oriented response: {goal_response}
        Rapport-building response: {rapport_response}
        
        The combined response should:
        1. Maintain the main objective from the goal-oriented response
        2. Incorporate rapport-building elements from the second response
        3. Feel natural and conversational
        4. Be concise and to the point
        
        Current free energy level: {self.current_free_energy}
        Current beliefs: {self.beliefs}
        
        Adjust the balance between goal-oriented and rapport-building elements based on the current free energy level and beliefs.
        
        Provide the combined response:
        """
        combined_response = self.goal_seeker.llm_provider.generate_response(prompt)
        logger.debug(f"Combined response: {combined_response}")
        return combined_response

    def _update_conversation_history(self, user_input: str, agent_response: str):
        self.conversation_history.append({"user": user_input, "agent": agent_response})
        # Limit history to last 10 exchanges to manage memory
        self.conversation_history = self.conversation_history[-10:]
        logger.debug(
            f"Updated conversation history. Current length: {len(self.conversation_history)}"
        )

    def run(self):
        logger.info(
            "LLM Proactive Agent is ready. Type 'exit' to end the conversation."
        )
        while True:
            user_input = input("User: ")
            if user_input.lower() == "exit":
                logger.info("Conversation ended by user")
                print("Agent: Goodbye!")
                break
            response = self.process_user_input(user_input)
            print(f"Agent: {response}")

    def set_initial_goal(self, goal: str):
        logger.info(f"Setting initial goal: {goal}")
        self.goal_seeker.set_goal(goal)
        self.current_free_energy = self.goal_seeker.minimize_free_energy()
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
        # This is a placeholder for a method to visualize the goal hierarchy
        # You might want to implement this using a library like matplotlib or networkx
        logger.info("Goal Hierarchy Visualization:")
        for i, goal in enumerate(self.get_goal_hierarchy()):
            logger.info(f"{i+1}. {goal}")

    def handle_proactive_behavior(self):
        logger.info("Handling proactive behavior")
        # This method could be called periodically to allow the agent to take proactive actions
        # based on its current state, goals, and beliefs
        if self.current_free_energy > self.proactive_threshold:
            new_goal = self.goal_seeker._generate_potential_goals()[0]  # Just an example
            self.set_initial_goal(new_goal)
            logger.info(f"Proactively set new goal: {new_goal}")
        # Add more proactive behaviors as needed
