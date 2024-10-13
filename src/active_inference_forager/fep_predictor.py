from active_inference_forager.user_model import UserModel
from active_inference_forager.config import PREDICTION_THRESHOLD

class FEPPredictor:
    def __init__(self, user_model: UserModel):
        self.user_model = user_model

    def predict_next_topic(self) -> str:
        preferences = self.user_model.get_preferences()
        if not preferences:
            return "general"
        predicted_topic = max(preferences, key=preferences.get)
        if preferences[predicted_topic] >= PREDICTION_THRESHOLD:
            return predicted_topic
        return "general"
