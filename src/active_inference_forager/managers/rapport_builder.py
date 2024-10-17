from textblob import TextBlob

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
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        if polarity < -0.6:
            return "very_negative"
        elif -0.6 <= polarity < -0.2:
            return "negative"
        elif -0.2 <= polarity < 0.2:
            if subjectivity < 0.4:
                return "neutral"
            else:
                return "mixed"
        elif 0.2 <= polarity < 0.6:
            return "positive"
        else:
            return "very_positive"

    def generate_response(self, action, user_input):
        emotion = self.analyze_sentiment(user_input)
        print(f"Detected emotion: {emotion}")
        if action == "empathetic_response":
            if emotion == "very_negative":
                prompt = f"The user said '{user_input}' and seems very upset. Respond with strong empathy and offer support."
            elif emotion == "negative":
                prompt = f"The user said '{user_input}' and  appears to be feeling down. Respond with empathy and encouragement."
            elif emotion == "neutral":
                prompt = f"The user said '{user_input}' and  seems neutral. Respond in a friendly and supportive manner."
            elif emotion == "mixed":
                prompt = f"The user said '{user_input}' and their emotions seem mixed. Acknowledge their complex feelings and offer a balanced response."
            elif emotion == "positive":
                prompt = f"The user said '{user_input}' and seems to be in a good mood. Respond positively and build on their enthusiasm."
            elif emotion == "very_positive":
                prompt = f"The user said '{user_input}' and is very happy. Share in their excitement and reinforce their positive feelings."
        else:
            prompt = f"Assist the user based on their input: '{user_input}'. Their emotional state seems to be {emotion}."

        response = self.llm_provider.generate_response(prompt)
        return response
