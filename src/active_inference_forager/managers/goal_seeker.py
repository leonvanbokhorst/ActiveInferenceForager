import spacy
import numpy as np
from typing import List, Dict, Any
from textblob import TextBlob
from active_inference_forager.managers.interaction_manager import InteractionManager
from active_inference_forager.providers.llm_provider import LLMProvider
from active_inference_forager.models.llm_inference_engine import LLMInferenceEngine


class GoalSeeker(InteractionManager):
    def __init__(self, inference_engine: LLMInferenceEngine, llm_provider: LLMProvider):
        super().__init__(inference_engine)
        self.llm_provider = llm_provider
        self.current_goal = None
        self.goal_hierarchy = []
        self.nlp = spacy.load("en_core_web_sm")

    def set_goal(self, goal: str):
        self.current_goal = goal

    def handle_proactive_behavior(self, user_input: str) -> str:
        pass

    def process_input(self, user_input: str) -> str:
        # Process user input and generate a response
        observations = self.extract_features(user_input)
        beliefs = self.inference_engine.infer(observations)
        action = self.inference_engine.choose_action(beliefs)
        response = self.generate_response(action, user_input, observations)
        return response

    def extract_features(self, user_input: str) -> Dict[str, Any]:
        doc = self.nlp(user_input)

        # Extract entities
        entities = [(ent.text, ent.label_) for ent in doc.ents]

        # Extract key phrases (noun chunks)
        key_phrases = [chunk.text for chunk in doc.noun_chunks]

        # Extract sentiment
        blob = TextBlob(user_input)
        sentiment = blob.sentiment.polarity

        # Extract main verbs (actions)
        actions = [token.lemma_ for token in doc if token.pos_ == "VERB"]

        # Extract dependencies
        dependencies = [(token.text, token.dep_, token.head.text) for token in doc]

        # Relevance to current goal
        goal_relevance = self.calculate_goal_relevance(user_input)

        return {
            "entities": entities,
            "key_phrases": key_phrases,
            "sentiment": sentiment,
            "actions": actions,
            "dependencies": dependencies,
            "goal_relevance": goal_relevance,
        }

    def calculate_goal_relevance(self, user_input: str) -> float:
        if not self.current_goal:
            return 0.0

        # Use the LLM to calculate relevance
        prompt = f"""
        Given the current goal: "{self.current_goal}"
        And the user input: "{user_input}"
        
        On a scale of 0 to 1, how relevant is the user input to the current goal?
        Provide only a number as the response.
        """

        relevance_score = float(self.llm_provider.generate_response(prompt))
        return min(
            max(relevance_score, 0.0), 1.0
        )  # Ensure the score is between 0 and 1

    def generate_response(
        self, action: str, user_input: str, observations: Dict[str, Any]
    ) -> str:
        # Prepare context for response generation
        context = self._prepare_context(action, observations)

        # Generate response using LLM
        prompt = self._create_response_prompt(context, user_input)
        response = self.llm_provider.generate_response(prompt)

        # Update internal state based on the generated response
        self._update_internal_state(response)

        return response

    def _prepare_context(
        self, action: str, observations: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "current_goal": self.current_goal,
            "goal_hierarchy": self.goal_hierarchy,
            "chosen_action": action,
            "extracted_features": observations,
            "predicted_outcome": self._predict_outcome(action, observations),
        }

    def _create_response_prompt(self, context: Dict[str, Any], user_input: str) -> str:
        return f"""
        Current goal: {context['current_goal']}
        Goal hierarchy: {context['goal_hierarchy']}
        Chosen action: {context['chosen_action']}
        Extracted features: {context['extracted_features']}
        Predicted outcome: {context['predicted_outcome']}
        User input: {user_input}

        Given the above context, generate a response that:
        1. Aligns with the current goal and chosen action
        2. Addresses relevant extracted features from the user input
        3. Moves towards the predicted outcome
        4. Maintains coherence with the goal hierarchy

        Ensure the response is natural and conversational. Do not explicitly mention the goals or extracted features in the response.
        """

    def _predict_outcome(self, action: str, observations: Dict[str, Any]) -> str:
        # Use the LLM to predict the outcome of the chosen action
        prompt = f"""
        Given the current goal: "{self.current_goal}"
        Chosen action: "{action}"
        And the following observations:
        {observations}

        Predict the likely outcome of taking this action. Provide a brief, one-sentence description.
        """
        return self.llm_provider.generate_response(prompt)

    def _update_internal_state(self, response: str):
        # Update goal relevance based on the generated response
        self.goal_relevance = self.calculate_goal_relevance(response)

        # If the goal relevance is low, consider updating the goal
        if self.goal_relevance < 0.3:  # This threshold can be adjusted
            self.update_goal_hierarchy()

    def update_goal_hierarchy(self):
        # Evaluate current goal
        current_goal_value = self._evaluate_goal(self.current_goal)

        # Generate potential new goals
        potential_goals = self._generate_potential_goals()

        # Evaluate potential goals
        goal_values = [self._evaluate_goal(goal) for goal in potential_goals]

        # Update goal hierarchy
        self._update_hierarchy(potential_goals, goal_values)

        # Set new current goal
        self.current_goal = self.goal_hierarchy[0] if self.goal_hierarchy else None

    def _evaluate_goal(self, goal: str) -> float:
        prompt = f"""
        Given the current situation and the goal: "{goal}"
        Evaluate the following aspects on a scale of 0 to 1:
        1. Relevance: How relevant is this goal to the current situation?
        2. Achievability: How achievable is this goal given the current circumstances?
        3. Value: How valuable would achieving this goal be?
        4. Urgency: How urgent is this goal?

        Provide your evaluation as four numbers separated by commas, representing the scores for relevance, achievability, value, and urgency respectively.
        """
        response = self.llm_provider.generate_response(prompt)
        scores = [float(score.strip()) for score in response.split(",")]
        return np.mean(scores)  # Simple average of all scores

    def _generate_potential_goals(self) -> List[str]:
        prompt = f"""
        Given the current goal: "{self.current_goal}"
        And the current situation, generate 3 potential new goals that could be relevant.
        These goals should be related to the current goal but may be more specific, more general, or adjacent goals that could be valuable to pursue.
        Provide each goal on a new line.
        """
        response = self.llm_provider.generate_response(prompt)
        return [goal.strip() for goal in response.split("\n") if goal.strip()]

    def _update_hierarchy(self, potential_goals: List[str], goal_values: List[float]):
        # Combine current goals and potential new goals
        all_goals = self.goal_hierarchy + potential_goals
        all_values = [
            self._evaluate_goal(goal) for goal in self.goal_hierarchy
        ] + goal_values

        # Sort goals by their values
        sorted_goals = [x for _, x in sorted(zip(all_values, all_goals), reverse=True)]

        # Update goal hierarchy, keeping top 5 goals
        self.goal_hierarchy = sorted_goals[:5]

    def _calculate_free_energy(self, goal: str) -> float:
        # Simplified free energy calculation
        # In a more complex implementation, this would involve comparing predicted and actual outcomes
        expected_outcome = self._predict_outcome(
            self.inference_engine.choose_action({}), {}
        )
        actual_outcome = self._evaluate_current_state()
        return abs(
            float(expected_outcome) - float(actual_outcome)
        )  # BUG: actual_outcome is a string, should be a float

    def _evaluate_current_state(self) -> float:
        prompt = f"""
        Given the current goal: "{self.current_goal}"
        Evaluate the current state on a scale of 0 to 1, where 0 means the current state is very far from the goal, and 1 means the goal has been achieved.
        Provide only a number as the response.
        """
        return float(self.llm_provider.generate_response(prompt))

    def minimize_free_energy(self):
        # Calculate free energy for current goal and top alternative
        current_free_energy = self._calculate_free_energy(self.current_goal)
        alternative_goal = (
            self.goal_hierarchy[1]
            if len(self.goal_hierarchy) > 1
            else self.current_goal
        )
        alternative_free_energy = self._calculate_free_energy(alternative_goal)

        # If alternative goal has lower free energy, switch to it
        if alternative_free_energy < current_free_energy:
            self.current_goal = alternative_goal
            self.update_goal_hierarchy()  # Reorganize hierarchy based on new current goal
