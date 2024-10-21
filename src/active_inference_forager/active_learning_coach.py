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
        self.inferred_goals: Dict[str, float] = {}
        self.conversation_history: List[Dict] = []
        self.current_topic: str = ""
        self.confidence_level: float = 0.5
        self.vector_store = EnhancedSemanticVectorStore()
        self.recent_interactions: List[Dict] = []
        self.last_api_call_time = 0
        self.api_cooldown = 1
        self.belief_state: Dict[str, float] = {}
        self.previous_precision = 1.0
        self.goal_decay_rate = 0.95
        self.user_state = {
            "topic": "",
            "understanding_level": 0.5,
            "engagement": 0.8,
            "emotional_state": "neutral",
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
        self.update_inferred_goals(user_input, system_response)
        self.update_user_state(user_input, system_response)

    def update_confidence_level(self, user_input: str):
        similar_texts = self.vector_store.find_similar(user_input, k=2)
        if len(similar_texts) > 1:
            similarity = 1 - similar_texts[1][1] / 2
            self.confidence_level = (self.confidence_level + similarity) / 2
        else:
            self.confidence_level = max(0.1, self.confidence_level - 0.1)

    def update_inferred_goals(self, user_input: str, system_response: str):
        for goal in self.inferred_goals:
            self.inferred_goals[goal] *= self.goal_decay_rate

        new_goals = self.infer_goals_from_input(user_input)
        for goal in new_goals:
            if goal in self.inferred_goals:
                self.inferred_goals[goal] = min(self.inferred_goals[goal] + 0.2, 1.0)
            else:
                self.inferred_goals[goal] = 0.5

        total_prob = sum(self.inferred_goals.values())
        if total_prob > 0:
            for goal in self.inferred_goals:
                self.inferred_goals[goal] /= total_prob

        self.inferred_goals = {k: v for k, v in self.inferred_goals.items() if v > 0.05}

    def infer_goals_from_input(self, user_input: str) -> List[str]:
        goals = []
        keywords = {
            "learn": "wants to learn about",
            "know": "wants to know about",
            "understand": "wants to understand",
            "explain": "wants an explanation of",
            "curious": "is curious about",
            "help": "needs help with",
            "sad": "is feeling sad about",
            "angry": "is feeling angry about",
            "confused": "is confused about",
        }
        for keyword, goal_prefix in keywords.items():
            if keyword in user_input.lower():
                goals.append(f"The user {goal_prefix} {self.current_topic}")

        if not goals:
            goals.append(f"The user wants to explore {self.current_topic}")

        return goals

    def update_belief_state(self, user_input: str, system_response: str):
        keywords = [
            "confused",
            "understand",
            "clear",
            "difficult",
            "easy",
            "interesting",
            "boring",
        ]
        for keyword in keywords:
            if keyword in user_input.lower():
                self.belief_state[keyword] = self.belief_state.get(keyword, 0) + 1

        for goal, probability in self.inferred_goals.items():
            self.belief_state[goal] = probability

        total = sum(self.belief_state.values())
        if total > 0:
            self.belief_state = {k: v / total for k, v in self.belief_state.items()}

    def update_user_state(self, user_input: str, system_response: str):
        # Update topic
        self.user_state["topic"] = self.current_topic

        # Update understanding level
        if "understand" in user_input.lower() or "clear" in user_input.lower():
            self.user_state["understanding_level"] = min(
                1.0, self.user_state["understanding_level"] + 0.1
            )
        elif "confused" in user_input.lower() or "difficult" in user_input.lower():
            self.user_state["understanding_level"] = max(
                0.0, self.user_state["understanding_level"] - 0.1
            )

        # Update engagement
        if "interesting" in user_input.lower() or "tell me more" in user_input.lower():
            self.user_state["engagement"] = min(
                1.0, self.user_state["engagement"] + 0.1
            )
        elif "boring" in user_input.lower() or "bye" in user_input.lower():
            self.user_state["engagement"] = max(
                0.0, self.user_state["engagement"] - 0.1
            )

        # Update emotional state
        emotional_keywords = {
            "happy": "positive",
            "excited": "positive",
            "sad": "negative",
            "angry": "negative",
            "frustrated": "negative",
            "confused": "neutral",
        }
        for keyword, state in emotional_keywords.items():
            if keyword in user_input.lower():
                self.user_state["emotional_state"] = state
                break

    def get_belief_state_summary(self) -> str:
        return ", ".join([f"{k}: {v:.2f}" for k, v in self.belief_state.items()])

    def get_top_goals(self, n: int = 3) -> List[Tuple[str, float]]:
        return sorted(self.inferred_goals.items(), key=lambda x: x[1], reverse=True)[:n]


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
    top_goals = state.get_top_goals()
    goals_str = "\n".join([f"- {goal}: {prob:.2f}" for goal, prob in top_goals])

    prompt = f"""Context:
    {context}

    User input: {user_input}

    Top inferred goals:
    {goals_str}

    User state:
    Topic: {state.user_state['topic']}
    Understanding level: {state.user_state['understanding_level']:.2f}
    Engagement: {state.user_state['engagement']:.2f}
    Emotional state: {state.user_state['emotional_state']}

    Based on the context, user input, inferred goals, and user state, perform the following tasks:
    1. Respond to the user's input, addressing their most likely goals and emotional state.
    2. If the goals are vague or the topic has changed, ask a clarifying question.
    3. Provide a response that either explores the user's knowledge or explains the next concept, depending on the understanding level and engagement.
    4. Include a brief explanation or fact related to the topic in your response.
    5. If the user's emotional state is negative, offer support or encouragement.
    6. Adjust your language complexity based on the user's understanding level.

    Format your response as:
    Response: [your response to the user]
    """

    response = generate_system_response(prompt)
    state.update_state(user_input, response)
    return response


def detect_topic_llm(
    user_input: str, current_topic: str, conversation_history: List[Dict]
) -> str:
    recent_history = conversation_history[-3:]  # Get the last 3 interactions
    conversation_context = "\n".join(
        [
            f"User: {inter['user']}\nSystem: {inter['system']}"
            for inter in recent_history
        ]
    )

    prompt = f"""Given the following conversation context and the user's latest input, identify the main topic of discussion. If the topic has changed, provide the new topic. If it's a continuation of the previous topic, state that topic.

Conversation context:
{conversation_context}

User's latest input: {user_input}

Current topic: {current_topic}

Respond with only the topic name, or 'Same topic' if it hasn't changed."""

    try:
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            messages=[
                {
                    "role": "system",
                    "content": "You are a expert topic detection assistant.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=50,
        )
        detected_topic = response.choices[0].message.content.strip()
        return current_topic if detected_topic == "Same topic" else detected_topic
    except Exception as e:
        print(f"Error in topic detection: {str(e)}")
        return current_topic


def process_user_input(user_input: str, state: EnhancedConversationState) -> str:
    context = get_relevant_context(state, user_input)
    top_goals = state.get_top_goals()
    goals_str = "\n".join([f"- {goal}: {prob:.2f}" for goal, prob in top_goals])

    prompt = f"""Context:
    {context}

    User input: {user_input}

    Current topic: {state.current_topic}

    Top inferred goals:
    {goals_str}

    User state:
    Topic: {state.user_state['topic']}
    Understanding level: {state.user_state['understanding_level']:.2f}
    Engagement: {state.user_state['engagement']:.2f}
    Emotional state: {state.user_state['emotional_state']}

    Based on the context, user input, current topic, inferred goals, and user state, perform the following tasks:
    1. Respond to the user's input, addressing their most likely goals and emotional state.
    2. If the topic has changed, acknowledge the new topic and ask a relevant question to explore the user's interest.
    3. Provide a response that either explores the user's knowledge or explains the next concept, depending on the understanding level and engagement.
    4. If the user's emotional state is negative, offer support or encouragement.
    5. Adjust your language complexity based on the user's understanding level.
    6. Keep your response short, conversational, and engaging.

    Format your response as:
    Response: [your response to the user]
    """

    response = generate_system_response(prompt)
    state.update_state(user_input, response)
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

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Thank you for learning with me today! Goodbye!")
            break

        state.current_topic = detect_topic_llm(
            user_input, state.current_topic, state.conversation_history
        )

        response = process_user_input(user_input, state)

        interaction_count += 1
        print(f"---\nInteraction {interaction_count}\n---\nCoach: {response.strip()}")
        print(f"---\nCurrent confidence level: {state.confidence_level:.2f}")
        print(f"Current topic: {state.current_topic}")
        print(f"Inferred goals: {state.get_top_goals()}")
        print(f"Belief state: {state.get_belief_state_summary()}")
        print(f"User state: {state.user_state}")


if __name__ == "__main__":
    main_interaction_loop()
