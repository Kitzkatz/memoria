from core.logger import debug, info, error
import requests
from cache.config import settings


class LLMAdapter:

    def chat(self, prompt: str):
        """
        Send a prompt to the LLM and return the response text.

        Args:
            prompt: The prompt to send

        Returns:
            str: The LLM response text

        Raises:
            requests.exceptions.RequestException: If the request fails
            KeyError: If the response is malformed
        """
        url = f"{settings.LLM_URL}{settings.LLM_ENDPOINT}"
        debug(f"LLM request to {url}", category="llm")

        payload = {
            "prompt": prompt,
            "max_tokens": settings.LLM_MAX_TOKENS,
            "temperature": settings.LLM_TEMPERATURE,
            "repeat_penalty": 1.1
        }

        # Add stop tokens if configured
        if hasattr(settings, "LLM_STOP_TOKENS") and settings.LLM_STOP_TOKENS:
            payload["stop"] = settings.LLM_STOP_TOKENS

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=settings.LLM_TIMEOUT
            )
            response.raise_for_status()

            # Only log response details if debug is enabled
            info(f"LLM response status: {response.status_code}")
            debug(f"LLM raw response: {response.text[:200]}...", category="llm")

            data = response.json()

            # Handle different response formats
            if "choices" in data and len(data["choices"]) > 0:
                # OpenAI-style response
                choice = data["choices"][0]
                if "text" in choice:
                    text = choice["text"]
                elif "message" in choice and "content" in choice["message"]:
                    text = choice["message"]["content"]
                else:
                    error(f"Unexpected response format: {choice}")
                    return ""
            elif "response" in data:
                # Simpler API format
                text = data["response"]
            else:
                error(f"Unknown response format: {data.keys()}")
                return ""

            if not text or not text.strip():
                error("LLM returned empty response")
                return ""

            debug(f"LLM response: {text[:100]}...", category="llm")
            return text.strip()

        except requests.exceptions.Timeout:
            debug(f"LLM request timed out after {settings.LLM_TIMEOUT}s")
            return ""
        except requests.exceptions.RequestException as e:
            debug(f"LLM request failed: {e}")
            return ""
        except KeyError as e:
            error(f"Unexpected LLM response structure: {e}")
            return ""
        except Exception as e:
            debug(f"LLM adapter error: {e}")
            return ""

    def chat_with_history(self, prompt: str, history: list = None):
        """
        Send a prompt with conversation history.

        Args:
            prompt: The prompt to send
            history: List of previous messages (optional)

        Returns:
            str: The LLM response text
        """
        if history is None:
            history = []

        # Build conversation context
        context = ""
        for msg in history:
            context += f"{msg.get('role', 'user')}: {msg.get('content', '')}\n"

        context += f"user: {prompt}\nassistant:"

        return self.chat(context)

    def stream_chat(self, prompt: str):
        """
        Stream response from the LLM (if supported).

        Returns:
            generator: Yields chunks of the response
        """
        url = f"{settings.LLM_URL}{settings.LLM_ENDPOINT}"
        payload = {
            "prompt": prompt,
            "max_tokens": settings.LLM_MAX_TOKENS,
            "temperature": settings.LLM_TEMPERATURE,
            "stream": True
        }

        try:
            response = requests.post(
                url,
                json=payload,
                stream=True,
                timeout=settings.LLM_TIMEOUT
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    decoded = line.decode('utf-8')
                    if decoded.startswith('data: '):
                        yield decoded[6:]

        except Exception as e:
            error(f"LLM stream error: {e}")
            yield f"[ERROR: {e}]"
