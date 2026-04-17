"""
genome.py
---------
GraphGenome: the unit of evolution.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GraphGenome:
    """
    Wraps a graph specification with evolutionary metadata.

    Attributes
    ----------
    spec      : Graph definition with ``nodes`` and ``edges`` lists.
    origin    : Human-readable label describing how this genome was created.
    parents   : List of parent specs (empty for the initial population).
    fitness   : Accumulated fitness score; reset each generation.
    last_output: The execution output dict captured during the last battle.
    """

    spec: dict
    origin: str = "initial"
    parents: list[dict] = field(default_factory=list)
    fitness: float = 0.0
    last_output: dict = field(default_factory=dict)

    def reset_fitness(self) -> None:
        """Zero out the fitness score before a new generation's battles."""
        self.fitness = 0.0

    def to_snapshot(self) -> dict[str, Any]:
        """Serialise to a plain dict for history recording / UI display."""
        return {
            "spec": self.spec,
            "fitness": self.fitness,
            "origin": self.origin,
            "parents": self.parents,
            "output": self.last_output,
        }
