import os
from active_inference_forager.llm_proactive_agent import LLMProactiveAgent
from active_inference_forager.models.llm_generative_model import LLMGenerativeModel
from active_inference_forager.models.llm_inference_engine import LLMInferenceEngine
from active_inference_forager.providers.openai_provider import OpenAIProvider
from active_inference_forager.managers.rapport_builder import RapportBuilder
from active_inference_forager.managers.goal_seeker import GoalSeeker

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


def main():
    try:
        # Initialize the LLM provider
        llm_provider = OpenAIProvider()

        # Initialize components
        generative_model = LLMGenerativeModel(llm_provider)
        inference_engine = LLMInferenceEngine(generative_model)
        goal_seeker = GoalSeeker(inference_engine, llm_provider)

        # Create the agent
        agent = LLMProactiveAgent(goal_seeker)

        # Set initial goal
        initial_goal = "Flirt with the user and make them feel special."
        agent.set_initial_goal(initial_goal)

        # Main interaction loop
        print("Type 'exit' to end the conversation.")
        while True:
            user_input = input("User: ")
            if user_input.lower() == "exit":
                print("Thank you for the conversation. Goodbye!")
                break

            # Process user input and get response
            response = agent.process_user_input(user_input)
            print(f"\nAgent: {response}")

            # Log current state after each interaction

            # Visualize goal hierarchy (placeholder)
            agent.visualize_goal_hierarchy()

            # Display conversation summary
            conversation_summary = agent._summarize_conversation_history()

            # Handle proactive behavior
            agent.handle_proactive_behavior()

    except ValueError as ve:
        logger.error(f"ValueError: {str(ve)}")
        print(f"Error: {str(ve)}")
        print("Please set your OpenAI API key as an environment variable:")
        print("export OPENAI_API_KEY='your-api-key-here'")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        print(f"An unexpected error occurred: {str(e)}")


if __name__ == "__main__":
    main()
