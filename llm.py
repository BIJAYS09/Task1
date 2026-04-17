"""

Thin wrapper around the Groq client with retry logic.
"""

import time
from groq import Groq
import config


_client: Groq | None = None


def _get_client() -> Groq:
    """Lazy-initialise the Groq client so tests can patch config first."""
    global _client
    if _client is None:
        _client = Groq(api_key=config.GROQ_API_KEY)
    return _client


def call(system: str, user: str, temp: float = 0.7, retries: int = 3) -> str:
    """
    Call the LLM with a system + user prompt.

    Returns the text response, or an empty string on repeated failure.
    """
    config.increment_llm_calls()
    client = _get_client()

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=config.MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temp,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            wait = 2 ** attempt          # exponential back-off: 1 s, 2 s, 4 s
            if attempt < retries - 1:
                time.sleep(wait)
            else:
                print(f"[llm] all {retries} retries failed: {exc}")

    return ""
