"""
Post-execution reflection: asks the LLM to critique an agent's output
and return a list of improvement suggestions.
"""

import llm as llm
from utils import extract_json, truncate


_SYSTEM = "You are a critical reviewer of AI reasoning chains."

_PROMPT_TEMPLATE = """
Analyze the output below and identify concrete improvements.

Task:
{task}

Output:
{output}

Respond ONLY with valid JSON in exactly this shape:
{{"improvements": ["improvement 1", "improvement 2"]}}
""".strip()


def reflect_on_output(output: dict, task: str) -> dict:
    """
    Ask the LLM to critique *output* with respect to *task*.

    Returns a dict with an ``"improvements"`` key (list of strings),
    or an empty dict if parsing fails.
    """
    prompt = _PROMPT_TEMPLATE.format(task=task, output=truncate(output, 1500))
    raw = llm.call(_SYSTEM, prompt, temp=0.4)
    return extract_json(raw) or {}
