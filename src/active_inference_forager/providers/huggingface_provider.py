from transformers import pipeline


class HuggingFaceProvider(LLMProvider):
    def __init__(self, model_name="gpt2"):
        self.generator = pipeline("text-generation", model=model_name)

    def generate_response(self, prompt, **kwargs):
        outputs = self.generator(
            prompt,
            max_length=kwargs.get("max_tokens", 150),
            num_return_sequences=1,
            temperature=kwargs.get("temperature", 0.7),
        )
        return outputs[0]["generated_text"]
