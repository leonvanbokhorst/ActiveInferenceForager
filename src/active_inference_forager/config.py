# Configuration settings for the Active Inference Forager

# LLM settings
LLM_MODEL = "mock"  # Default LLM model to use (currently only 'mock' is supported)

# Conversation settings
MAX_CONVERSATION_HISTORY = 10  # Maximum number of messages to keep in conversation history

# User model settings
DEFAULT_PREFERENCE_SCORE = 0.5  # Default score for new topics in user preferences
USER_MODEL_FILE = "user_model.json"  # File to save/load user model data

# FEP Predictor settings
PREDICTION_THRESHOLD = 0.7  # Threshold for considering a topic as predicted

# Logging settings
LOG_LEVEL = "INFO"  # Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_FILE = "active_inference_forager.log"  # File to save log messages

# File paths
USER_MODEL_PATH = "user_model.json"  # Path to save/load user model data

# AI Assistant settings
AI_NAME = "FEP Assistant"  # Name of the AI assistant
WELCOME_MESSAGE = "Welcome to the FEP-based Conversational AI. Type 'exit' to end the conversation."
GOODBYE_MESSAGE = "Thank you for using the FEP-based Conversational AI. Goodbye!"

# Error messages
ERROR_MESSAGE = "I'm sorry, but I encountered an error while processing your input. Please try again."
