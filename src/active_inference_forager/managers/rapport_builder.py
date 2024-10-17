from active_inference_forager.managers.interaction_manager import InteractionManager

class RapportBuilder(InteractionManager):
    def __init__(self, inference_engine, llm_provider):
        super().__init__(inference_engine)
        self.llm_provider = llm_provider

    def process_input(self, user_input):
        observations = self.extract_features(user_input)
        beliefs = self.inference_engine.infer(observations)
        action = self.inference_engine.choose_action(beliefs)
        response = self.generate_response(action, user_input)
        return response

    def handle_proactive_behavior(self):
        # Additional proactive behaviors can be implemented here
        pass

    def extract_features(self, user_input):
        # Use NLP tools to extract features
        emotion = self.analyze_sentiment(user_input)
        return {"emotion": emotion}

    def analyze_sentiment(self, text):
        from textblob import TextBlob

        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        if polarity < -0.3:
            return "frustrated"
        elif polarity > 0.3:
            return "happy"
        else:
            return "neutral"

    def generate_response(self, action, user_input):
        if action == "empathetic_response":
            prompt = f"You notice the user is frustrated. Respond empathetically to help them feel understood."
        else:
            prompt = f"Assist the user based on their input: '{user_input}'"

        response = self.llm_provider.generate_response(prompt)
        return response
