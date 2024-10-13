# Active Inference Forager

## Project Overview
The Active Inference Forager is a conversational AI assistant that leverages the principles of Active Inference to engage in natural language interactions with users. The goal of this project is to create a foundation for a conversational AI system that can adapt to user preferences and evolve its communication strategies over time.

## Architecture
The project follows a modular and scalable architecture, with the following key components:

- **ActiveInferenceAgent**: Responsible for processing user input, generating responses, and managing the overall conversation flow.
- **ConversationManager**: Maintains the conversation history and provides methods for adding and retrieving messages.
- **FEPPredictor**: Implements the basic Free Energy Principle (FEP)-based prediction logic to anticipate user inputs and minimize surprise.
- **LLMInterface**: Defines the contract for interacting with the Large Language Model (LLM) used for content generation.
- **UserModel**: Manages the user's preferences and state, which are essential for adaptive learning and personalization.
- **FeedbackManager**: Collects user feedback, which can be used to inform future improvements and adaptations.
- **DataManager**: Provides a flexible data storage solution, currently using MongoDB, to support the evolving data requirements of the project.

## Development Environment Setup
1. Install Python 3.12 (or the latest version)
2. Create a virtual environment and activate it
3. Install the project dependencies: `pip install -r requirements.txt`
4. Run the unit tests: `pytest tests/unit`
5. Start the application: `python src/active_inference_forager/main.py`

## Contributing
We welcome contributions to the Active Inference Forager project. Please follow the guidelines outlined in the [CONTRIBUTING.md](CONTRIBUTING.md) file.

## Roadmap
This project is currently in stage 1, which focuses on laying the foundation for the FEP-based conversational framework. Future stages will build upon this foundation, incorporating more advanced features such as:

- Stage 2: Active Inference in User Modeling
- Stage 3: Adaptive Learning and Personalization
- Stage 4: Multi-Modal Interaction and Embodied Cognition

For more details on the development plan, please refer to the [development-plan.md](/docs/development-plan.md) file.
