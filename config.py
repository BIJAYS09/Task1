"""
config.py
---------
Central configuration and shared mutable state.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Model ──────────────────────────────────────────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
MODEL: str = "meta-llama/llama-4-scout-17b-16e-instruct"

# ── Graph limits ───────────────────────────────────────────────────────────────
MAX_NODES: int = 6

# ── Shared counters (mutated at runtime) ──────────────────────────────────────
LLM_CALLS: int = 0


def reset_llm_calls() -> None:
    global LLM_CALLS
    LLM_CALLS = 0


def increment_llm_calls() -> None:
    global LLM_CALLS
    LLM_CALLS += 1
