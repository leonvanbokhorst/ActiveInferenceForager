import json


class LLMInferenceEngine:
    def __init__(self, generative_model):
        self.generative_model = generative_model
        self.belief_state = {}
        self.memory = []  # Log of previous interactions

    def rapid_belief_update(self, user_input, sentiment):
        urgent_update_needed = sentiment < -0.5 or any(
            phrase in user_input.lower()
            for phrase in ["stop", "leave me alone", "not interested"]
        )
        if urgent_update_needed:
            self.belief_state = {
                "user_comfort": "low",
                "conversation_state": "needs_termination",
                "priority": "respect_boundaries",
            }

    def infer(self, observations):
        self.memory.append(observations)
        self.update_beliefs(observations)
        return self.belief_state

    def update_beliefs(self, observations):
        prompt = self._create_belief_update_prompt(observations)
        response = self.generative_model.predict(prompt)
        try:
            updated_beliefs = response
            self.belief_state.update(updated_beliefs)
        except json.JSONDecodeError:
            print("Error: Unable to parse LLM response as JSON.")

    def choose_action(self, beliefs):
        prompt = self._create_action_selection_prompt(beliefs)
        return self.generative_model.predict(prompt)

    def _create_belief_update_prompt(self, observations):
        return f"""
        Given the following observations and current beliefs, update the belief state:

        Observations: {observations}
        Current Beliefs: {self.belief_state}

        Please return an updated belief state as a JSON object. Include probabilities for relevant states.
        """

    def _create_action_selection_prompt(self, beliefs):
        return f"""
        Given the following belief state, choose the most appropriate action:

        Belief State: {beliefs}

        Please return a single action as a string.
        """
