"""
Mutation strategies that produce new GraphGenome instances from existing ones.
"""

import random

import llm as llm
import config
from genome import GraphGenome
from utils import clean_id, extract_json


# ── Prompts ────────────────────────────────────────────────────────────────────

_MUTATE_SYSTEM = "You are a creative AI graph architect."

_MUTATE_PROMPT = """
Design a DIFFERENT reasoning graph for the task below.
Be creative — vary node count, purposes, and approach.

Task:
{task}

Respond ONLY with valid JSON in exactly this shape:
{{"nodes": [{{"purpose": "...", "prompt": "..."}}]}}
""".strip()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_spec_from_nodes(raw_nodes: list[dict], parent_spec: dict) -> dict:
    """Convert raw LLM-produced node dicts into a validated graph spec."""
    nodes: list[dict] = []
    for i, n in enumerate(raw_nodes[: config.MAX_NODES]):
        purpose = n.get("purpose", f"Step {i}")
        prompt = n.get("prompt", f"Perform: {purpose}")
        nodes.append({"id": clean_id(purpose, i), "purpose": purpose, "prompt": prompt})

    edges = [[nodes[i]["id"], nodes[i + 1]["id"]] for i in range(len(nodes) - 1)]
    return {"nodes": nodes, "edges": edges}


# ── Public strategies ──────────────────────────────────────────────────────────

def random_shuffle_mutation(spec: dict) -> GraphGenome:
    """
    Shuffle the existing nodes into a random order.

    Fast, deterministic alternative when the LLM fails to return valid JSON.
    """
    nodes = spec["nodes"][:]
    random.shuffle(nodes)
    edges = [[nodes[i]["id"], nodes[i + 1]["id"]] for i in range(len(nodes) - 1)]
    return GraphGenome(
        spec={"nodes": nodes, "edges": edges},
        origin="mutation_random_shuffle",
        parents=[spec],
    )


def llm_mutation(genome: GraphGenome, task: str) -> GraphGenome:
    """
    Ask the LLM to produce a creatively different graph for *task*.

    Falls back to :func:`random_shuffle_mutation` if the LLM response
    cannot be parsed or contains no nodes.
    """
    prompt = _MUTATE_PROMPT.format(task=task)
    raw = llm.call(_MUTATE_SYSTEM, prompt, temp=1.2)
    parsed = extract_json(raw)

    if parsed and isinstance(parsed.get("nodes"), list) and parsed["nodes"]:
        try:
            new_spec = _build_spec_from_nodes(parsed["nodes"], genome.spec)
            return GraphGenome(
                spec=new_spec,
                origin="mutation_llm",
                parents=[genome.spec],
            )
        except Exception:  # noqa: BLE001
            pass

    # Fallback
    return random_shuffle_mutation(genome.spec)


# ── Convenience alias ──────────────────────────────────────────────────────────

def mutate(genome: GraphGenome, task: str) -> GraphGenome:
    """Primary mutation entry-point used by the evolution engine."""
    return llm_mutation(genome, task)
