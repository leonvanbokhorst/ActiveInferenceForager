import logging
from typing import Optional
import argparse
from active_inference_forager.config import *
from active_inference_forager.llm_interface import LLMInterface, MockLLM
from active_inference_forager.active_inference_agent import ActiveInferenceAgent

# Set up logging
logging.basicConfig(filename=LOG_FILE, level=LOG_LEVEL)
logger = logging.getLogger(__name__)


def main(model: Optional[str] = None) -> None:
    if model == "mock" or model is None:
        llm = MockLLM()
    else:
        raise ValueError(f"Unsupported model: {model}")

    agent = ActiveInferenceAgent(llm)
    agent.load_user_model(USER_MODEL_PATH)

    print(WELCOME_MESSAGE)

    try:
        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                break

            response = agent.process_user_input(user_input)
            print(f"AI: {response}")

            # Print some debug information
            predicted_topic = agent.fep_predictor.predict_next_topic()
            confidence = agent.fep_predictor.get_prediction_confidence()
            print(
                f"Debug: Predicted topic: {predicted_topic}, Confidence: {confidence:.2f}"
            )

    except KeyboardInterrupt:
        print("\nConversation interrupted by user.")
    finally:
        agent.save_user_model(USER_MODEL_PATH)
        print(GOODBYE_MESSAGE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the FEP-based Conversational AI")
    parser.add_argument(
        "--model", type=str, default="mock", help="The LLM model to use (default: mock)"
    )
    args = parser.parse_args()

    main(args.model)
