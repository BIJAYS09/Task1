"""
main.py
-------
Command-line entry point for the Stem Agent Evolution system.

Usage:
    python -m stem_agent.main
    # or
    python stem_agent/main.py
"""

import config
from compiler import GraphCompiler
from evolution import Evolution
from utils import summarize_spec


def run(task: str, generations: int = 4) -> None:
    config.reset_llm_calls()

    print(f"\nTask : {task}")
    print(f"Generations: {generations}\n")

    evo = Evolution(task)
    best = evo.evolve(generations=generations)

    print("\n" + "=" * 60)
    print("EVOLUTION SUMMARY")
    print("=" * 60)
    print("Initial agent :", summarize_spec(evo.initial))
    print("Final   agent :", summarize_spec(best.spec))

    print("\n" + "=" * 60)
    print("FINAL EXECUTION OUTPUT")
    print("=" * 60)
    result = GraphCompiler(best.spec).run(task)
    for node_id, output in result.get("data", {}).items():
        print(f"\n[{node_id}]\n{output}")

    print("\n" + "=" * 60)
    print(f"Total LLM calls: {config.LLM_CALLS}")


if __name__ == "__main__":
    task = input("Enter task: ").strip()
    if not task:
        task = "Explain the concept of entropy in simple terms."
    run(task)
