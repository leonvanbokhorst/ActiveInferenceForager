import os
import time
import numpy as np
import faiss
from typing import List, Dict, Tuple, Optional
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class EnhancedSemanticVectorStore:
    def __init__(self, vector_dimension=1536, nlist=100, use_gpu=False):
        self.vector_dimension = vector_dimension
        self.texts = []
        self.use_gpu = use_gpu

        quantizer = faiss.IndexFlatL2(vector_dimension)
        self.index = faiss.IndexIVFFlat(
            quantizer, vector_dimension, nlist, faiss.METRIC_L2
        )

        if use_gpu:
            self.res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(self.res, 0, self.index)

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        model = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
        response = client.embeddings.create(input=texts, model=model)
        return np.array([data.embedding for data in response.data], dtype=np.float32)

    def add_texts(self, texts: List[str]):
        if not self.index.is_trained:
            print("Training the index...")
            dummy_vectors = np.random.rand(
                max(self.index.nlist, len(texts)), self.vector_dimension
            ).astype("float32")
            self.index.train(dummy_vectors)

        vectors = self.get_embeddings(texts)
        self.index.add(vectors)
        self.texts.extend(texts)

    def find_similar(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        if not self.texts:
            return []
        query_vector = self.get_embeddings([query])
        D, I = self.index.search(query_vector, min(k, len(self.texts)))
        return [
            (self.texts[i], float(D[0][j]))
            for j, i in enumerate(I[0])
            if i < len(self.texts)
        ]


class EnhancedConversationState:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.inferred_goals: List[str] = []
        self.conversation_history: List[Dict] = []
        self.current_topic: str = ""
        self.confidence_level: float = 0.5
        self.vector_store = EnhancedSemanticVectorStore()
        self.recent_interactions: List[Dict] = []
        self.last_api_call_time = 0
        self.api_cooldown = 1
        self.belief_state: Dict[str, float] = {}
        self.previous_precision = 1.0

        # Initialize user_state with the required fields
        self.user_state = {
            "topic": "",
            "understanding_level": 0.5,  # between 0 and 1
            "engagement": 0.8,  # between 0 and 1
        }

    def update_state(self, user_input: str, system_response: str):
        self.conversation_history.append(
            {"user": user_input, "system": system_response}
        )
        self.vector_store.add_texts([user_input])
        self.update_confidence_level(user_input)
        self.recent_interactions.append({"user": user_input, "system": system_response})
        if len(self.recent_interactions) > 5:
            self.recent_interactions.pop(0)
        self.update_belief_state(user_input, system_response)

    def update_confidence_level(self, user_input: str):
        similar_texts = self.vector_store.find_similar(user_input, k=2)
        if len(similar_texts) > 1:
            similarity = 1 - similar_texts[1][1] / 2
            self.confidence_level = (self.confidence_level + similarity) / 2

    def update_belief_state(self, user_input: str, system_response: str):
        keywords = ["confused", "understand", "clear", "difficult"]
        for keyword in keywords:
            if keyword in user_input.lower():
                self.belief_state[keyword] = self.belief_state.get(keyword, 0) + 1

        for goal in self.inferred_goals:
            self.belief_state[goal] = self.belief_state.get(goal, 0) + 0.5

    def get_belief_state_summary(self) -> str:
        return ", ".join([f"{k}: {v:.2f}" for k, v in self.belief_state.items()])


# Enhanced decision-making using Expected Free Energy, precision modulation, and novelty bonus


def generative_model(user_state: Dict, action: str) -> float:
    base_improvement = 0
    if action == "explain concept":
        base_improvement = 0.2
    elif action == "provide example":
        base_improvement = 0.1
    elif action == "ask question":
        base_improvement = 0.15

    uncertainty = np.random.normal(0, 0.05)
    return user_state["understanding_level"] + base_improvement + uncertainty


def precision_modulation(
    predicted: float, actual: float, previous_precision: float
) -> float:
    error = abs(predicted - actual)
    new_precision = 1 / (error + 0.1)
    return 0.8 * previous_precision + 0.2 * new_precision


def expected_free_energy(
    user_state: Dict, action: str, precision: float = 1.0
) -> float:
    epistemic_value = 1.0 if action == "ask question" else 0.5
    predicted_understanding = generative_model(user_state, action)
    pragmatic_value = (
        predicted_understanding - user_state["understanding_level"]
    ) * precision
    novelty_bonus = np.random.uniform(0.0, 0.1) if action == "recommend resource" else 0
    beta = 0.5
    G = pragmatic_value + beta * epistemic_value + novelty_bonus
    return G


def select_action(
    user_state: Dict, coach_actions: List[str], precision: float = 1.0
) -> str:
    best_action = None
    max_G = float("-inf")

    for action in coach_actions:
        G = expected_free_energy(user_state, action, precision)
        if G > max_G:
            max_G = G
            best_action = action

    return best_action


def generate_system_response(prompt: str) -> str:
    try:
        model = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an intelligent Active Learning Coach, designed to help users learn and understand various topics...",
                },
                {"role": "user", "content": prompt},
            ],
            stream=True,
        )
        full_response = ""
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                full_response += chunk.choices[0].delta.content
        return full_response
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return "I'm sorry, I'm having trouble generating a response right now."


def get_relevant_context(state: EnhancedConversationState, user_input: str) -> str:
    recent_interactions = "\n".join(
        [
            f"User: {i['user']}\nSystem: {i['system']}"
            for i in state.recent_interactions[-3:]
        ]
    )
    relevant_info = state.vector_store.find_similar(user_input, k=2)
    relevant_context = "\n".join([f"Related: {text}" for text, _ in relevant_info])
    return f"""
    Recent interactions:
    {recent_interactions}

    Relevant information:
    {relevant_context}

    Current topic: {state.current_topic}
    Current confidence level: {state.confidence_level:.2f}
    Belief state: {state.get_belief_state_summary()}
    """


def process_user_input(user_input: str, state: EnhancedConversationState) -> str:
    context = get_relevant_context(state, user_input)
    prompt = f"""Context:
    {context}

    User input: {user_input}

    Based on the context and user input, perform the following tasks:
    1. Infer the user's learning goals.
    2. If the goals are vague, generate a clarifying question.
    3. If the goals are clear, provide a response that either explores the user's knowledge or explains the next concept, depending on the confidence level.
    4. Include a brief explanation or fact related to the topic in your response.
    5. Consider the current belief state and adjust your response accordingly.

    Format your response as:
    Goals: [inferred goals]
    Response: [your response to the user]
    """

    current_time = time.time()
    if current_time - state.last_api_call_time < state.api_cooldown:
        time.sleep(state.api_cooldown - (current_time - state.last_api_call_time))
    response = generate_system_response(prompt)
    state.last_api_call_time = time.time()
    return response


def main_interaction_loop():
    if not os.getenv("OPENAI_API_KEY"):
        print(
            "Error: OPENAI_API_KEY is not set. Please set it as an environment variable."
        )
        return

    state = EnhancedConversationState("user123")
    print(
        "Welcome to your Enhanced Active Learning Coach! What would you like to learn about today?"
    )

    interaction_count = 0
    coach_actions = [
        "explain concept",
        "provide example",
        "ask question",
        "recommend resource",
    ]

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Thank you for learning with me today! Goodbye!")
            break

        # Generate a prediction using the generative model
        predicted = generative_model(state.user_state, "ask question")

        # Simulate an actual outcome based on user feedback (for illustration)
        actual = predicted + np.random.normal(0, 0.05)

        # Update precision based on prediction error
        state.previous_precision = precision_modulation(
            predicted, actual, state.previous_precision
        )

        # Select the action with the highest Expected Free Energy
        action = select_action(
            state.user_state, coach_actions, state.previous_precision
        )

        # Generate a response using OpenAI's GPT
        response = process_user_input(user_input, state)
        goals, coach_response = response.split("Response:", 1)

        # Update inferred goals and the current topic
        state.inferred_goals = goals.replace("Goals:", "").strip().split(", ")
        state.current_topic = (
            state.inferred_goals[0] if state.inferred_goals else state.current_topic
        )
        interaction_count += 1

        # Print the coach's response
        print(
            f"---\nInteraction {interaction_count}\n---\nCoach: {coach_response.strip()}"
        )

        # Update the state with the user's input and the coach's response
        state.update_state(user_input, coach_response.strip())

        # Display current confidence, goals, and belief state
        print(f"---\nCurrent confidence level: {state.confidence_level:.2f}")
        print(f"Inferred goals: {state.inferred_goals}")
        print(f"Belief state: {state.get_belief_state_summary()}")


if __name__ == "__main__":
    main_interaction_loop()
