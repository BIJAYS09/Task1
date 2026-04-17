"""
Core evolutionary loop: population management, battles, and selection.
"""

import random

import llm as llm
from compiler import GraphCompiler
from genome import GraphGenome
from mutation import mutate, random_shuffle_mutation
from reflection import reflect_on_output
from scoring import diversity_bonus, empty_penalty, structure_score
from tasks import strategy_to_graph
from utils import extract_json, truncate

# ── Battle ─────────────────────────────────────────────────────────────────────

_JUDGE_SYSTEM = "You are an objective judge of AI reasoning quality."

_JUDGE_PROMPT = """
Compare the two agent outputs below and score each from 1–10.

Agent A:
{out_a}

Agent B:
{out_b}

Respond ONLY with valid JSON:
{{"scoreA": <int>, "scoreB": <int>}}
""".strip()


def battle(a: GraphGenome, b: GraphGenome, task: str) -> None:
    """
    Run both genomes against *task*, score them, and update fitness in-place.

    Fitness contributions:
    - LLM judge score (1–10 each)
    - Structure heuristic bonus/penalty
    - Diversity bonus (unique node purposes × 0.3)
    - Reflection improvement count × 0.5
    - Empty-output penalty (−2 per near-empty node)
    - Small random noise (±0.3) for exploration
    """
    out_a = GraphCompiler(a.spec).run(task)
    out_b = GraphCompiler(b.spec).run(task)

    a.last_output = out_a
    b.last_output = out_b

    # LLM judge
    prompt = _JUDGE_PROMPT.format(out_a=truncate(out_a, 1000), out_b=truncate(out_b, 1000))
    scores = extract_json(llm.call(_JUDGE_SYSTEM, prompt, temp=0)) or {}

    a.fitness += float(scores.get("scoreA", 1))
    b.fitness += float(scores.get("scoreB", 1))

    # Structural heuristics
    a.fitness += structure_score(a.spec)
    b.fitness += structure_score(b.spec)

    # Diversity bonus
    a.fitness += diversity_bonus(a.spec) * 0.3
    b.fitness += diversity_bonus(b.spec) * 0.3

    # Reflection bonus
    a.fitness += len(reflect_on_output(out_a, task).get("improvements", [])) * 0.5
    b.fitness += len(reflect_on_output(out_b, task).get("improvements", [])) * 0.5

    # Empty-output penalty
    a.fitness -= empty_penalty(out_a)
    b.fitness -= empty_penalty(out_b)

    # Noise (keeps exploration alive)
    a.fitness += random.uniform(-0.3, 0.3)
    b.fitness += random.uniform(-0.3, 0.3)


# ── Evolution engine ───────────────────────────────────────────────────────────

_POPULATION_SIZE = 4


class Evolution:
    """
    Manages a fixed-size population of :class:`GraphGenome` instances,
    evolves them over multiple generations, and tracks history.

    Parameters
    ----------
    task : str
        The natural-language task all genomes are evaluated on.
    """

    def __init__(self, task: str) -> None:
        self.task = task
        base = strategy_to_graph(task)
        self.initial: dict = base

        self.population: list[GraphGenome] = [
            GraphGenome(base, origin="base"),
            random_shuffle_mutation(base),
            GraphGenome(strategy_to_graph("Creative approach: " + task), origin="creative"),
            GraphGenome(
                spec={
                    "nodes": [
                        {
                            "id": "understand_0",
                            "purpose": "Understand",
                            "prompt": "Understand the task deeply.",
                        },
                        {
                            "id": "answer_1",
                            "purpose": "Answer",
                            "prompt": "Provide a direct, concrete answer.",
                        },
                    ],
                    "edges": [["understand_0", "answer_1"]],
                },
                origin="minimal",
            ),
        ]

        self.history: list[list[dict]] = []

    # ------------------------------------------------------------------
    def _run_generation(self) -> None:
        """Reset fitness, battle pairs, sort, snapshot, breed next gen."""
        for genome in self.population:
            genome.reset_fitness()

        # Battle consecutive pairs
        for i in range(0, len(self.population) - 1, 2):
            battle(self.population[i], self.population[i + 1], self.task)

        # Sort descending by fitness
        self.population.sort(key=lambda g: g.fitness, reverse=True)

        # Record snapshot
        self.history.append([g.to_snapshot() for g in self.population])

        # Breed next generation: keep top-2, fill rest with mutations
        survivors = self.population[:2]
        children: list[GraphGenome] = []

        while len(survivors) + len(children) < _POPULATION_SIZE:
            parent = random.choice(self.population)
            child = mutate(parent, self.task)
            child.parents = [parent.spec]
            children.append(child)

        self.population = survivors + children

    # ------------------------------------------------------------------
    def evolve(self, generations: int = 4) -> GraphGenome:
        """
        Run *generations* rounds of evolution and return the best genome.

        If the best genome is identical to the initial spec (no evolution
        occurred), returns the last genome instead as a fallback.
        """
        for gen in range(generations):
            print(f"\n=== Generation {gen} ===")
            self._run_generation()
            print(f"  Top fitness: {self.population[0].fitness:.2f}")

        best = self.population[0]

        # Fallback: prefer something that actually changed
        if best.spec == self.initial and len(self.population) > 1:
            best = self.population[-1]

        return best
