from active_inference_forager.llm_proactive_agent import LLMProactiveAgent
from active_inference_forager.models.simple_generative_model import SimpleGenerativeModel
from active_inference_forager.models.simpel_inference_engine import SimpleInferenceEngine
from active_inference_forager.providers.openai_provider import OpenAIProvider
from active_inference_forager.managers.rapport_builder import RapportBuilder
from active_inference_forager.managers.goal_seeker import GoalSeeker

# When instantiating the agent
llm_provider = OpenAIProvider(api_key="YOUR_API_KEY", model="gpt-4")

# Initialize inference engine and interaction managers with the LLM provider
generative_model = SimpleGenerativeModel()
inference_engine = SimpleInferenceEngine(generative_model)
rapport_builder = RapportBuilder(inference_engine, llm_provider)
goal_seeker = GoalSeeker(inference_engine, llm_provider)

# Create the agent
agent = LLMProactiveAgent(rapport_builder, goal_seeker)

if __name__ == "__main__":
    # Run the agent
    agent.run()
