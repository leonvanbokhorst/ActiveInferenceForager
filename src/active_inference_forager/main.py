import logging
from typing import Dict, Any
import numpy as np
from scipy.special import softmax

# TODO: Import actual LLM library when decided
from unittest.mock import MagicMock

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Placeholder for LLM
llm = MagicMock()

# Simplified set of possible user intents
INTENTS = ["greeting", "question", "statement", "farewell"]


def initialize_state() -> Dict[str, Any]:
    """
    Initialize the conversation state with prior probabilities for user intents.
    """
    return {
        "intent_probs": np.ones(len(INTENTS)) / len(INTENTS),  # Uniform prior
        "last_input": "",
        "surprise_history": [],
    }


def update_intent_probabilities(probs: np.ndarray, surprise: float) -> np.ndarray:
    """
    Update intent probabilities based on surprise.
    """
    # Increase probability of unlikely intents (high surprise)
    updated_probs = probs * (1 + surprise)
    return softmax(updated_probs)


def calculate_surprise(predicted_intent: str, actual_intent: str) -> float:
    """
    Calculate surprise based on predicted and actual intent.
    """
    return 0 if predicted_intent == actual_intent else 1


def predict_intent(user_input: str) -> str:
    """
    Predict the user's intent based on their input.
    This is a placeholder function and should be replaced with a more sophisticated model.
    """
    user_input = user_input.lower()
    # Expanded list of greeting phrases
    greetings = [
        "hello",
        "hi",
        "hey",
        "greetings",
        "good morning",
        "good afternoon",
        "good evening",
    ]
    farewells = ["bye", "goodbye", "see you", "farewell", "take care"]

    if any(greeting in user_input for greeting in greetings):
        return "greeting"
    elif any(farewell in user_input for farewell in farewells):
        return "farewell"
    elif user_input.endswith("?"):
        return "question"
    else:
        return "statement"


def process_input(
    user_input: str, conversation_state: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Process user input using FEP principles.
    """
    logger.info(f"Processing input: {user_input}")

    # Predict intent
    predicted_intent = INTENTS[np.argmax(conversation_state["intent_probs"])]
    actual_intent = predict_intent(user_input)

    # Calculate surprise
    surprise = calculate_surprise(predicted_intent, actual_intent)
    conversation_state["surprise_history"].append(surprise)

    # Update intent probabilities
    conversation_state["intent_probs"] = update_intent_probabilities(
        conversation_state["intent_probs"], surprise
    )

    conversation_state["last_input"] = user_input
    conversation_state["last_intent"] = actual_intent

    logger.info(f"Predicted intent: {predicted_intent}, Actual intent: {actual_intent}")
    logger.info(f"Surprise: {surprise}")
    logger.info(f"Updated intent probabilities: {conversation_state['intent_probs']}")

    return conversation_state


def generate_response(conversation_state: Dict[str, Any]) -> str:
    """
    Generate a response using the LLM based on the conversation state.
    """
    logger.info("Generating response")

    # TODO: Implement actual LLM-based response generation
    # For now, we'll use a simple rule-based response
    intent = conversation_state["last_intent"]
    if intent == "greeting":
        response = "Hello! How can I assist you today?"
    elif intent == "question":
        response = "That's an interesting question. Let me think about it."
    elif intent == "statement":
        response = "I understand. Can you tell me more about that?"
    elif intent == "farewell":
        response = "Goodbye! It was nice talking to you."
    else:
        response = "I'm not sure I understand. Could you please rephrase that?"

    return f"AI: {response}"


def main():
    logger.info("Starting FEP-Based Conversational Framework")
    conversation_state = initialize_state()

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit", "bye"]:
                logger.info("Ending conversation")
                break

            conversation_state = process_input(user_input, conversation_state)
            response = generate_response(conversation_state)
            print(response)

        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")
            print("I'm sorry, I encountered an error. Could you please try again?")


if __name__ == "__main__":
    main()
