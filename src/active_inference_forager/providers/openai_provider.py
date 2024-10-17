import os

from openai import OpenAI
from active_inference_forager.providers.llm_provider import LLMProvider


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key=None, model="gpt-4"):
        if not api_key:
            # get from env
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("API key is required.")

        self.api_key = api_key
        self.model = model
        self.client = OpenAI(api_key=self.api_key)

    def generate_response(self, prompt, **kwargs):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": kwargs.get(
                        "system_prompt", "You are a helpful assistant."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=kwargs.get("max_tokens", 150),
            temperature=kwargs.get("temperature", 0.7),
        )
        return response.choices[0].message.content.strip()
