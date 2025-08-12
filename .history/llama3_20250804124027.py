# llama3_together.py

from together import Together

class Llama3TogetherWrapper:
    def __init__(self, api_key: str, model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"):
        self.client = Together(api_key=api_key)
        self.model = model

    def run(self, messages: list[dict]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        return response.choices[0].message.content
