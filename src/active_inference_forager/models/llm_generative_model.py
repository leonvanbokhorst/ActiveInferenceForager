import json
from typing import Dict, Any
from active_inference_forager.models.generative_model import GenerativeModel
from active_inference_forager.providers.llm_provider import LLMProvider


class LLMGenerativeModel(GenerativeModel):
    def __init__(self, llm_provider: LLMProvider):
        super().__init__()
        self.llm_provider = llm_provider
        self.prior_beliefs = {
            "intent": None,
            "emotion": None,
            "user_state": None,
            "conversation_context": None,
        }
        self.posterior_beliefs = self.prior_beliefs.copy()

    def predict(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        # Generate a prompt for the LLM to predict user state and intentions
        prompt = self._generate_prediction_prompt(observations)

        # Use the LLM to generate predictions
        llm_response = self.llm_provider.generate_response(prompt)

        try:
            # Parse the LLM response to extract predictions
            predictions = json.loads(llm_response)

        except json.JSONDecodeError:
            print("Error: Unable to parse LLM response as JSON.")

        return predictions

    def update_beliefs(self, observations: Dict[str, Any]):
        # Generate a prompt for the LLM to update beliefs
        prompt = self._generate_belief_update_prompt(observations)

        # Use the LLM to generate updated beliefs
        llm_response = self.llm_provider.generate_response(prompt)

        try:
            updated_beliefs = json.loads(llm_response)

            # Update posterior beliefs
            self.posterior_beliefs.update(updated_beliefs)
        except json.JSONDecodeError:
            print("Error: Unable to parse LLM response as JSON.")

    def _generate_prediction_prompt(self, observations: Dict[str, Any]) -> str:
        return f"""
        Given the following observations about a user interaction:
        {observations}
        
        Please predict the following:
        1. The user's likely intent
        2. The user's emotional state
        3. The user's overall state (e.g., confused, satisfied, frustrated)
        4. The current context of the conversation

        Provide your predictions in a structured format. No markdown. JSON only.
        """

    def _generate_belief_update_prompt(self, observations: Dict[str, Any]) -> str:
        return f"""
        Current beliefs:
        {self.posterior_beliefs}

        New observations:
        {observations}

        Please update the beliefs based on these new observations. Consider how the new information should influence our understanding of the user's intent, emotional state, overall state, and the conversation context.

        Provide the updated beliefs in a structured format. JSON only.
        """

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        # This is a simplified parser. In a real implementation, you'd want a more robust parsing method.
        parsed_response = {}
        for line in response.split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                parsed_response[key.strip().lower()] = value.strip()
        return parsed_response
