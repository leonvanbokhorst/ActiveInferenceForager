import spacy
import logging
from logging.handlers import RotatingFileHandler
from textblob import TextBlob
from transformers import pipeline

from active_inference_forager.managers.interaction_manager import InteractionManager

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


class RapportBuilder(InteractionManager):
    def __init__(self, inference_engine, llm_provider):
        super().__init__(inference_engine)
        self.llm_provider = llm_provider
        self.nlp = spacy.load("en_core_web_sm")
        self.emotion_classifier = pipeline(
            "text-classification",
            model="bhadresh-savani/distilbert-base-uncased-emotion",
            return_all_scores=True,
        )
        logger.info("RapportBuilder initialized")

    def process_input(self, user_input):
        logger.info(f"Processing user input: {user_input}")
        observations = self.extract_features(user_input)
        beliefs = self.inference_engine.infer(observations)
        action = self.inference_engine.choose_action(beliefs)
        logger.info(f"Chosen action: {action}")
        response = self.generate_response(action, user_input, observations)
        return response

    def handle_proactive_behavior(self):
        #logger.info("Handling proactive behavior")
        pass

    def extract_features(self, user_input):
        logger.info("Extracting features from user input")
        doc = self.nlp(user_input)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        dependencies = [(token.text, token.dep_, token.head.text) for token in doc]
        emotion = self.detect_emotion(user_input)
        sentiment = self.analyze_sentiment(user_input)
        logger.info(
            f"Extracted features: emotion={emotion}, sentiment={sentiment}, entities={entities}, dependencies={dependencies}"
        )
        return {
            "entities": entities,
            "dependencies": dependencies,
            "emotion": emotion,
            "sentiment": sentiment,
        }

    def detect_emotion(self, text):
        logger.info("Detecting emotion")
        results = self.emotion_classifier(text)[0]
        emotion = max(results, key=lambda x: x["score"])["label"]
        logger.info(f"Detected emotion: {emotion}")
        return emotion

    def analyze_sentiment(self, text):
        logger.info("Analyzing sentiment")
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        if polarity < -0.6:
            sentiment = "very_negative"
        elif -0.6 <= polarity < -0.2:
            sentiment = "negative"
        elif -0.2 <= polarity < 0.2:
            sentiment = "neutral" if subjectivity < 0.4 else "mixed"
        elif 0.2 <= polarity < 0.6:
            sentiment = "positive"
        else:
            sentiment = "very_positive"

        logger.info(f"Analyzed sentiment: {sentiment}")
        return sentiment

    def generate_response(self, action, user_input, observations):
        logger.info("Generating response")
        emotion = observations["emotion"]
        sentiment = observations["sentiment"]
        entities = observations["entities"]

        system_prompt = """
        You are an AI assistant designed to build rapport with users. Your responses should be empathetic, 
        considerate of the user's emotional state, and relevant to the entities they mention. Adapt your 
        language and tone to match the user's sentiment and emotional state.
        """

        user_prompt = f"""
        User input: '{user_input}'
        Emotional state: {emotion}
        Sentiment: {sentiment}
        Entities mentioned: {entities}

        Action to take: {'Provide an empathetic response' if action == 'empathetic_response' else 'Assist the user'}

        Generate a response that:
        1. Acknowledges the user's emotional state
        2. Addresses the entities mentioned if relevant
        3. Provides assistance or empathy based on the specified action
        """

        logger.info(f"RapportBuilder: Generating response for action '{action}'")
        #Slogger.debug(f"RapportBuilder: User Prompt: {user_prompt}")

        response = self.llm_provider.generate_response(
            user_prompt, system_prompt=system_prompt
        )
        logger.info(f"Generated response: {response}")
        return response
