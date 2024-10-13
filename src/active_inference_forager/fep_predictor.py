import numpy as np
from active_inference_forager.user_model import UserModel
from active_inference_forager.config import PREDICTION_THRESHOLD


class FEPPredictor:
    def __init__(self, user_model: UserModel):
        self.user_model = user_model
        self.prediction_history = []

    def predict_next_topic(self) -> str:
        preferences = self.user_model.get_preferences()
        if not preferences:
            return "general"

        # Calculate prediction probabilities
        total_preference = sum(preferences.values())
        probabilities = {
            topic: pref / total_preference for topic, pref in preferences.items()
        }

        # Add some exploration factor
        exploration_factor = 0.1
        for topic in probabilities:
            probabilities[topic] = (1 - exploration_factor) * probabilities[
                topic
            ] + exploration_factor / len(probabilities)

        # Make prediction
        predicted_topic = max(probabilities, key=probabilities.get)
        self.prediction_history.append(predicted_topic)

        # Calculate prediction error
        actual_topic = self.user_model.get_last_topic()
        prediction_error = 0 if predicted_topic == actual_topic else 1

        # Update model based on prediction error
        self.update_model(prediction_error)

        return predicted_topic

    def update_model(self, prediction_error: float):
        learning_rate = 0.1
        for topic in self.user_model.get_preferences():
            current_pref = self.user_model.get_preferences()[topic]
            if topic == self.user_model.get_last_topic():
                new_pref = current_pref + learning_rate * (1 - current_pref)
            else:
                new_pref = current_pref + learning_rate * (0 - current_pref)
            self.user_model.update_preference(topic, new_pref)

    def get_prediction_confidence(self) -> float:
        preferences = self.user_model.get_preferences()
        if not preferences:
            return 0.0
        return max(preferences.values()) / sum(preferences.values())
