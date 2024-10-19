import spacy
import numpy as np
import re
from typing import List, Dict, Any, Tuple
from textblob import TextBlob
from active_inference_forager.managers.interaction_manager import InteractionManager
from active_inference_forager.providers.llm_provider import LLMProvider
from active_inference_forager.models.llm_inference_engine import LLMInferenceEngine

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


class GoalSeeker(InteractionManager):
    def __init__(self, inference_engine: LLMInferenceEngine, llm_provider: LLMProvider):
        super().__init__(inference_engine)
        self.llm_provider = llm_provider
        self.current_goal = None
        self.goal_hierarchy = []
        self.nlp = spacy.load("en_core_web_sm")
        self.recent_energy_changes = []

    def set_goal(self, goal: str):
        self.current_goal = goal

    def handle_proactive_behavior(self):
        if self.goal_relevance < 0.3:
            self.update_goal_hierarchy()
            self.minimize_free_energy()

    def process_input(self, user_input: str) -> str:
        observations = self.extract_features(user_input)
        beliefs = self.inference_engine.infer(observations)
        action = self.inference_engine.choose_action(beliefs)
        response = self.generate_response(action, user_input, observations)
        return response

    def extract_features(self, user_input: str) -> Dict[str, Any]:
        doc = self.nlp(user_input)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        key_phrases = [chunk.text for chunk in doc.noun_chunks]
        blob = TextBlob(user_input)
        sentiment = blob.sentiment.polarity
        actions = [token.lemma_ for token in doc if token.pos_ == "VERB"]
        dependencies = [(token.text, token.dep_, token.head.text) for token in doc]
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
        prompt = f"""
        Given the current goal: "{self.current_goal}"
        And the user input: "{user_input}"
        
        On a scale of 0 to 1, how relevant is the user input to the current goal?
        Provide only a number as the response.
        """
        relevance_score = float(self.llm_provider.generate_response(prompt))
        logger.info(f"Calculated goal relevance: {relevance_score}")
        return min(max(relevance_score, 0.0), 1.0)

    def generate_response(
        self, action: str, user_input: str, observations: Dict[str, Any]
    ) -> str:
        context = self._prepare_context(action, observations)
        prompt = self._create_response_prompt(context, user_input)
        response = self.llm_provider.generate_response(prompt)
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
        prompt = f"""
        Given the current goal: "{self.current_goal}"
        Chosen action: "{action}"
        And the following observations:
        {observations}

        Predict the likely outcome of taking this action. Provide a brief, one-sentence description.
        """
        response = self.llm_provider.generate_response(prompt)
        logger.info(f"Predicted outcome: {response}")
        return response

    def _update_internal_state(self, response: str):
        self.goal_relevance = self.calculate_goal_relevance(response)
        if self.goal_relevance < 0.3:
            self.update_goal_hierarchy()

    def update_goal_hierarchy(self):
        current_goal_value = self._evaluate_goal(self.current_goal)
        potential_goals = self._generate_potential_goals()
        goal_values = [self._evaluate_goal(goal) for goal in potential_goals]
        self._update_hierarchy(potential_goals, goal_values)
        self.current_goal = self.goal_hierarchy[0] if self.goal_hierarchy else None
        logger.info(f"Updated goal hierarchy: {self.goal_hierarchy}")
        logger.info(f"Current goal: {self.current_goal}")

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
        avg_score = np.mean(scores)
        logger.info(f"Evaluation for goal '{goal}': {avg_score}")
        return avg_score

    def _generate_potential_goals(self) -> List[str]:
        prompt = f"""
        Given the current goal: "{self.current_goal}"
        And the current situation, generate 3 potential new goals that could be relevant.
        These goals should be related to the current goal but may be more specific, more general, or adjacent goals that could be valuable to pursue.
        Provide each goal on a new line.
        """
        response = self.llm_provider.generate_response(prompt)
        potential_goals = [
            goal.strip() for goal in response.split("\n") if goal.strip()
        ]
        logger.info(f"Generated potential goals: {potential_goals}")
        return potential_goals

    def _update_hierarchy(self, potential_goals: List[str], goal_values: List[float]):
        all_goals = self.goal_hierarchy + potential_goals
        all_values = [
            self._evaluate_goal(goal) for goal in self.goal_hierarchy
        ] + goal_values
        sorted_goals = [x for _, x in sorted(zip(all_values, all_goals), reverse=True)]
        self.goal_hierarchy = sorted_goals[:5]

    def _calculate_free_energy(self, goal: str) -> float:
        action = self.inference_engine.choose_action({})
        expected_outcome = self._predict_outcome(action, {})
        actual_outcome = self._evaluate_current_state()
        logger.info(f"Expected outcome: {expected_outcome}")
        logger.info(f"Actual outcome: {actual_outcome}")

        prediction_error = self._calculate_prediction_error(
            expected_outcome, actual_outcome
        )

        complexity = self._calculate_complexity()
        entropy = self._calculate_entropy()
        logger.info(f"Prediction error: {prediction_error}")
        logger.info(f"Complexity: {complexity}")
        logger.info(f"Entropy: {entropy}")

        free_energy = prediction_error + complexity + entropy
        logger.info(f"Free energy for goal '{goal}': {free_energy}")

        return free_energy

    def _calculate_prediction_error(self, expected: str, actual: float) -> float:
        try:
            expected_float = float(expected)
        except ValueError:
            expected_float = 0.5  # Default value if conversion fails

        # Ensure both expected and actual are scalar values
        if isinstance(actual, np.ndarray):
            actual = np.mean(actual)

        return np.square(expected_float - actual)

    def _calculate_complexity(self) -> float:
        prior = self._get_prior_belief()
        posterior = self._get_posterior_belief()

        # Ensure prior and posterior have the same shape
        max_len = max(len(prior), len(posterior))
        prior = np.pad(
            prior, (0, max_len - len(prior)), "constant", constant_values=(1e-10,)
        )
        posterior = np.pad(
            posterior,
            (0, max_len - len(posterior)),
            "constant",
            constant_values=(1e-10,),
        )

        return self._kl_divergence(posterior, prior)

    def _calculate_entropy(self) -> float:
        posterior = self._get_posterior_belief()
        return -np.sum(
            posterior * np.log(posterior + 1e-10)
        )  # Add small constant to avoid log(0)

    def _kl_divergence(self, p, q):
        # Ensure p and q have the same shape
        max_len = max(len(p), len(q))
        p = np.pad(p, (0, max_len - len(p)), "constant", constant_values=(1e-10,))
        q = np.pad(q, (0, max_len - len(q)), "constant", constant_values=(1e-10,))

        return np.sum(p * np.log((p + 1e-10) / (q + 1e-10)))

    def _get_prior_belief(self) -> np.array:
        prompt = f"""
        Given the current goal: "{self.current_goal}"
        Provide a probability distribution over possible outcomes before taking any action.
        Return the distribution as a list of probabilities that sum to 1, formatted as follows:
        0.XX, 0.YY, 0.ZZ, ...
        """
        response = self.llm_provider.generate_response(prompt)
        probabilities = re.findall(r"0\.\d+", response)
        if not probabilities:
            return np.array([1.0])  # Default to certainty if no probabilities found
        logger.info(f"Prior belief: {probabilities}")
        return np.array([float(p) for p in probabilities])

    def _get_posterior_belief(self) -> np.array:
        prompt = f"""
        Given the current goal: "{self.current_goal}"
        And the current state: {self._evaluate_current_state()}
        Provide an updated probability distribution over possible outcomes.
        Return the distribution as a list of probabilities that sum to 1, formatted as follows:
        0.XX, 0.YY, 0.ZZ, ...
        """
        response = self.llm_provider.generate_response(prompt)
        probabilities = re.findall(r"0\.\d+", response)
        if not probabilities:
            return np.array([1.0])  # Default to certainty if no probabilities found
        logger.info(f"Posterior belief: {probabilities}")
        return np.array([float(p) for p in probabilities])

    def _evaluate_current_state(self) -> float:
        prompt = f"""
        Given the current goal: "{self.current_goal}"
        Evaluate the current state on a scale of 0 to 1, where 0 means the current state is very far from the goal, and 1 means the goal has been achieved.
        Provide only a number as the response.
        """
        current_state = float(self.llm_provider.generate_response(prompt))
        logger.info(f"Evaluation of current state: {current_state}")
        return current_state

    def minimize_free_energy(self) -> Tuple[float, bool]:
        current_free_energy = self._calculate_free_energy(self.current_goal)
        alternative_goals = self.goal_hierarchy[:3]  # Consider top 3 alternative goals
        alternative_energies = [
            self._calculate_free_energy(goal) for goal in alternative_goals
        ]

        min_energy = min(alternative_energies + [current_free_energy])
        goal_changed = False

        if min_energy < current_free_energy:
            self.current_goal = alternative_goals[
                alternative_energies.index(min_energy)
            ]
            self.update_goal_hierarchy()
            goal_changed = True

        energy_change = current_free_energy - min_energy
        self.recent_energy_changes.append(energy_change)
        if len(self.recent_energy_changes) > 10:
            self.recent_energy_changes.pop(0)

        logger.info(f"Minimized free energy: {min_energy}")
        logger.info(f"Goal changed: {goal_changed}")

        return min_energy, goal_changed

    def get_current_free_energy(self) -> float:
        return self._calculate_free_energy(self.current_goal)

    def get_recent_energy_changes(self) -> List[float]:
        return self.recent_energy_changes
