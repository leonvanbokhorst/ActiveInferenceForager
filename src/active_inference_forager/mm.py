import openai
import os
from typing import Dict, List, Tuple

# Assuming you've set up your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")


class ConversationState:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.inferred_goals: List[str] = []
        self.conversation_history: List[Dict] = []
        self.current_topic: str = ""
        self.confidence_level: float = 0.5  # Start at medium confidence
        self.semantic_features: Dict[str, float] = {}

    def update_state(self, user_input: str, system_response: str):
        self.conversation_history.append(
            {"user": user_input, "system": system_response}
        )
        # Here we would update semantic features, confidence level, etc.
        # For now, we'll use placeholder functions
        self.update_semantic_features(user_input)
        self.update_confidence_level()

    def update_semantic_features(self, user_input: str):
        # Placeholder: In a real implementation, this would use
        # embeddings or other NLP techniques to update semantic features
        pass

    def update_confidence_level(self):
        # Placeholder: This would adjust confidence based on user responses
        pass


def generate_system_response(prompt: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-40-mini",
        messages=[
            {
                "role": "system",
                "content": "You are an intelligent Active Learning Coach, designed to help users learn and understand various topics through adaptive, personalized instruction.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message["content"]


def infer_learning_goals(
    user_input: str, state: ConversationState
) -> Tuple[List[str], str]:
    prompt = f"Based on the user's input: '{user_input}', and their conversation history: {state.conversation_history}, infer their learning goals. If the goals are vague, generate a clarifying question."
    response = generate_system_response(prompt)

    # For simplicity, we'll assume the model returns goals and a question separated by '|||'
    goals_and_question = response.split("|||")
    inferred_goals = goals_and_question[0].strip().split(", ")
    clarifying_question = (
        goals_and_question[1].strip() if len(goals_and_question) > 1 else ""
    )

    return inferred_goals, clarifying_question


def exploration_exploitation_logic(state: ConversationState) -> str:
    if state.confidence_level < 0.6:  # Threshold for exploration
        prompt = f"Generate an exploration question to better understand the user's knowledge of {state.current_topic}."
    else:
        prompt = (
            f"Provide the next step or concept in learning about {state.current_topic}."
        )

    return generate_system_response(prompt)


def main_interaction_loop():
    state = ConversationState("user123")  # Initialize with a user ID
    print(
        "Welcome to your Active Learning Coach! What would you like to learn about today?"
    )

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Thank you for learning with me today! Goodbye!")
            break

        # Infer learning goals
        inferred_goals, clarifying_question = infer_learning_goals(user_input, state)
        state.inferred_goals = inferred_goals

        if clarifying_question:
            print(f"Coach: {clarifying_question}")
            continue

        # Determine whether to explore or exploit
        response = exploration_exploitation_logic(state)

        print(f"Coach: {response}")

        # Update the conversation state
        state.update_state(user_input, response)


if __name__ == "__main__":
    main_interaction_loop()
