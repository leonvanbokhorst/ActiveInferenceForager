import logging
from active_inference_forager.conversation_manager import ConversationManager
from active_inference_forager.user_model import UserModel
from active_inference_forager.fep_predictor import FEPPredictor
from active_inference_forager.llm_interface import LLMInterface
from active_inference_forager.config import DEFAULT_PREFERENCE_SCORE

logger = logging.getLogger(__name__)


class ActiveInferenceAgent:
    def __init__(self, llm: LLMInterface):
        self.conversation_manager = ConversationManager()
        self.llm = llm
        self.user_model = UserModel()
        self.fep_predictor = FEPPredictor(self.user_model)

    def process_user_input(self, user_input: str) -> str:
        try:
            self.conversation_manager.add_message("user", user_input)
            predicted_topic = self.fep_predictor.predict_next_topic()
            confidence = self.fep_predictor.get_prediction_confidence()

            context = {
                "predicted_topic": predicted_topic,
                "confidence": confidence,
                "user_preferences": self.user_model.get_preferences(),
                "conversation_history": self.conversation_manager.get_conversation_history(),
            }

            if confidence < 0.6:  # If confidence is low, actively seek information
                prompt = f"The user said: '{user_input}'. Based on this, ask a question to determine if the topic is related to {predicted_topic} or to explore a new topic."
            else:
                prompt = f"Respond to '{user_input}' with a focus on the topic: {predicted_topic}"

            response = self.llm.generate_response(prompt, context)
            self.conversation_manager.add_message("assistant", response)

            # Update user model
            self.user_model.add_interaction(user_input, response, predicted_topic)
            self.user_model.update_preference(predicted_topic, DEFAULT_PREFERENCE_SCORE)

            return response
        except Exception as e:
            logger.error(f"Error processing user input: {str(e)}")
            return "I'm sorry, but I encountered an error while processing your input. Please try again."

    def save_user_model(self, filename: str) -> None:
        self.user_model.save_to_file(filename)

    def load_user_model(self, filename: str) -> None:
        self.user_model.load_from_file(filename)
