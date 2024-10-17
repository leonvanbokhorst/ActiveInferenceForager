from typing import List, Dict, Any
from active_inference_forager.managers.rapport_builder import RapportBuilder
from active_inference_forager.managers.goal_seeker import GoalSeeker
from active_inference_forager.providers.llm_provider import LLMProvider
from active_inference_forager.models.llm_generative_model import LLMGenerativeModel


class LLMProactiveAgent:
    def __init__(self, rapport_builder: RapportBuilder, goal_seeker: GoalSeeker):
        self.rapport_builder = rapport_builder
        self.goal_seeker = goal_seeker
        self.conversation_history: List[Dict[str, str]] = []
        self.current_state: Dict[str, Any] = {}

    def process_user_input(self, user_input: str) -> str:
        # Extract features from user input
        observations = self.goal_seeker.extract_features(user_input)

        # Update current state
        self.current_state.update(observations)

        # Minimize free energy and potentially update goals
        self.goal_seeker.minimize_free_energy()

        # Generate response
        response = self.goal_seeker.process_input(user_input)

        # Build rapport
        rapport_response = self.rapport_builder.process_input(user_input)

        # Combine goal-oriented response with rapport-building elements
        final_response = self._combine_responses(response, rapport_response)

        # Update conversation history
        self._update_conversation_history(user_input, final_response)

        return final_response

    def _combine_responses(self, goal_response: str, rapport_response: str) -> str:
        prompt = f"""
        Combine the following two responses into a single, coherent response:
        
        Goal-oriented response: {goal_response}
        Rapport-building response: {rapport_response}
        
        The combined response should:
        1. Maintain the main objective from the goal-oriented response
        2. Incorporate rapport-building elements from the second response
        3. Feel natural and conversational
        4. Be concise and to the point
        
        Provide the combined response:
        """
        return self.goal_seeker.llm_provider.generate_response(prompt)

    def _update_conversation_history(self, user_input: str, agent_response: str):
        self.conversation_history.append({"user": user_input, "agent": agent_response})
        # Limit history to last 10 exchanges to manage memory
        self.conversation_history = self.conversation_history[-10:]

    def run(self):
        print("LLM Proactive Agent is ready. Type 'exit' to end the conversation.")
        while True:
            user_input = input("User: ")
            if user_input.lower() == "exit":
                print("Agent: Goodbye!")
                break
            response = self.process_user_input(user_input)
            print(f"Agent: {response}")

    def set_initial_goal(self, goal: str):
        self.goal_seeker.set_goal(goal)

    def get_current_goal(self) -> str:
        return self.goal_seeker.current_goal

    def get_goal_hierarchy(self) -> List[str]:
        return self.goal_seeker.goal_hierarchy

    def get_current_state(self) -> Dict[str, Any]:
        return self.current_state
