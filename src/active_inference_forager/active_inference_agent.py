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
            prompt = f"Respond to '{user_input}' with a focus on the topic: {predicted_topic}"
            response = self.llm.generate_response(prompt)
            self.conversation_manager.add_message("assistant", response)
            self.user_model.update_preference(predicted_topic, DEFAULT_PREFERENCE_SCORE)
            return response
        except Exception as e:
            logger.error(f"Error processing user input: {str(e)}")
            return "I'm sorry, but I encountered an error while processing your input. Please try again."
