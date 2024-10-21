from transformers import pipeline
from active_inference_forager.providers.llm_provider import LLMProvider

class HuggingFaceProvider(LLMProvider):
    def __init__(self, model_name="gpt2"):
        # Initialize the text generation pipeline with the specified model
        self.generator = pipeline("text-generation", model=model_name)

    def generate_response(self, prompt, **kwargs):
        # Retrieve the system prompt from kwargs, or use a default if not provided
        system_prompt = kwargs.get("system_prompt", "You are a helpful assistant.")
        
        # Combine system prompt and user prompt into a single string
        # This format mimics the structure used in conversational AI models
        combined_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"

        # Generate text using the combined prompt
        outputs = self.generator(
            combined_prompt,
            # Adjust max_length to account for the length of the combined prompt
            max_length=kwargs.get("max_tokens", 150) + len(combined_prompt),
            num_return_sequences=1,
            temperature=kwargs.get("temperature", 0.7),
        )
        
        generated_text = outputs[0]["generated_text"]
        # Extract only the assistant's response by removing the original prompt
        response = generated_text[len(combined_prompt):].strip()
        return response
