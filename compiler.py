"""
Compiles a graph spec into a runnable LangGraph state-machine
and executes it against a task.
"""

import llm as llm
from langgraph.graph import END, StateGraph  # noqa: F401 (END kept for clarity)


# ── Internal helpers ───────────────────────────────────────────────────────────

def _make_node_fn(node: dict):
    """
    Return a LangGraph-compatible callable for a single reasoning node.

    The node receives the accumulated state, extracts the latest output,
    calls the LLM, and returns an updated state dict.
    """

    def fn(state: dict) -> dict:
        # Pull the most recent output from previous nodes
        data: dict = state.get("data", {})
        previous_values = list(data.values())
        previous = previous_values[-1] if previous_values else "None"

        prompt = f"""
{node['prompt']}

Previous Step Output:
{previous}

Produce your output for THIS step ONLY. Be concrete and specific.
""".strip()

        result = llm.call("You are a focused reasoning agent.", prompt)

        if not result or len(result.strip()) < 10:
            result = f"[No substantive output produced for: {node['purpose']}]"

        new_data = dict(data)
        new_data[node["id"]] = result

        return {"input": state["input"], "data": new_data}

    # Give the closure a unique __name__ so LangGraph can distinguish nodes
    fn.__name__ = node["id"]
    return fn


# ── Public class ───────────────────────────────────────────────────────────────

class GraphCompiler:
    """
    Builds and runs a LangGraph graph from a ``spec`` dict.

    Parameters
    ----------
    spec : dict
        A dict with ``nodes`` (list of node dicts) and ``edges``
        (list of ``[source_id, dest_id]`` pairs).
    """

    def __init__(self, spec: dict) -> None:
        self.spec = spec

    # ------------------------------------------------------------------
    def _build(self) -> object:
        """Compile the spec into a frozen LangGraph graph."""
        builder: StateGraph = StateGraph(dict)

        for node in self.spec["nodes"]:
            builder.add_node(node["id"], _make_node_fn(node))

        for source, dest in self.spec["edges"]:
            builder.add_edge(source, dest)

        if self.spec["nodes"]:
            builder.set_entry_point(self.spec["nodes"][0]["id"])

        return builder.compile()

    # ------------------------------------------------------------------
    def run(self, task: str) -> dict:
        """
        Execute the graph against *task*.

        Returns the final state dict, which contains:
        - ``"input"``  – the original task string
        - ``"data"``   – a mapping of node-id → node output text
        """
        graph = self._build()
        return graph.invoke({"input": task, "data": {}})
