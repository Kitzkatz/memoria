from core.logger import debug
import requests
from cache.config import settings


class LLMAdapter:

    def chat(self, prompt: str):
        debug("url = ", f"{settings.LLM_URL}{settings.LLM_ENDPOINT}", "LLMADAPTER made a call")
        response = requests.post(
            f"{settings.LLM_URL}{settings.LLM_ENDPOINT}",
            json={
                "prompt": prompt,
                "max_tokens": settings.LLM_MAX_TOKENS,
                "temperature": settings.LLM_TEMPERATURE,
##                "stop": ["<|eot_id|>", "\n<|start_header_id|>"],
                "repeat_penalty": 1.1
            },
            timeout=settings.LLM_TIMEOUT
        )
        response.raise_for_status()
        data = response.json()
        
        print("STATUS:", response.status_code)
        print("JSON:", data)

        text = data["choices"][0]["text"]

        print("TEXT:", repr(text))

        return text
