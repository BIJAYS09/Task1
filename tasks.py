"""
Task decomposition and initial graph-spec generation.
"""

import config
import llm as llm
from utils import clean_id, extract_json

# ── Prompts ────────────────────────────────────────────────────────────────────

_ANALYSE_SYSTEM = "You are a task-decomposition expert."

_ANALYSE_PROMPT = """
Break the following task into clear, sequential reasoning steps.

Task:
{task}

Respond ONLY with valid JSON in exactly this shape:
{{"steps": ["step 1", "step 2", "step 3"]}}
""".strip()

_NODE_PROMPT_TEMPLATE = """
You are responsible ONLY for: {purpose}

Task: {task}

Rules:
- Do NOT attempt to solve the full task.
- Build meaningfully on the previous step's output.
- Your output must be concrete and specific.
""".strip()


# ── Public API ─────────────────────────────────────────────────────────────────

def analyze_task(task: str) -> list[str]:
    """
    Ask the LLM to decompose *task* into ordered reasoning steps.

    Falls back to a sensible default on parse failure.
    """
    prompt = _ANALYSE_PROMPT.format(task=task)
    raw = llm.call(_ANALYSE_SYSTEM, prompt, temp=1.0)
    result = extract_json(raw)

    if isinstance(result, dict) and "steps" in result:
        steps = result["steps"]
        if isinstance(steps, list) and steps:
            return steps

    return ["Understand the problem", "Reason through it", "Formulate an answer"]


def strategy_to_graph(task: str) -> dict:
    """
    Turn a task string into a linear graph spec suitable for LangGraph.

    Each step from :func:`analyze_task` becomes one node, capped at
    ``config.MAX_NODES``.
    """
    steps = analyze_task(task)

    nodes: list[dict] = []
    for i, step in enumerate(steps[: config.MAX_NODES]):
        nodes.append(
            {
                "id": clean_id(step, i),
                "purpose": step,
                "prompt": _NODE_PROMPT_TEMPLATE.format(purpose=step, task=task),
            }
        )

    edges = [[nodes[i]["id"], nodes[i + 1]["id"]] for i in range(len(nodes) - 1)]

    return {"nodes": nodes, "edges": edges}
