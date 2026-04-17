"""
Heuristic fitness-scoring functions that don't require an LLM call.
"""


def diversity_bonus(spec: dict) -> float:
    """
    Reward graphs whose nodes cover many *distinct* purposes.

    Returns the number of unique purpose labels (used as a float bonus).
    """
    purposes = [n["purpose"] for n in spec.get("nodes", [])]
    return float(len(set(purposes)))


def structure_score(spec: dict) -> int:
    """
    Simple structural heuristic:

    +1  first node describes *defining* the problem
    +1  last node describes *evaluating* or *answering*
    -1  first node is merely *collecting* (too vague)
    -1  only two nodes and one is 'understand' (shallow graph)
    """
    purposes = [n["purpose"].lower() for n in spec.get("nodes", [])]
    score = 0

    if purposes:
        if "define" in purposes[0]:
            score += 1
        if "collect" in purposes[0]:
            score -= 1
        if "evaluate" in purposes[-1] or "answer" in purposes[-1]:
            score += 1

    if "understand" in purposes and len(purposes) <= 2:
        score -= 1

    return score


def empty_penalty(output: dict) -> int:
    """
    Penalise graphs where nodes produced little or no content.

    Each near-empty node value costs 2 fitness points.
    """
    data = output.get("data", {})
    return sum(2 for v in data.values() if not v or len(str(v).strip()) < 10)
