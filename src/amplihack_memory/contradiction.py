"""Contradiction detection between knowledge facts.

Detects when two facts about the same concept contain conflicting numerical
values, indicating an update or contradiction. Uses simple heuristics based
on shared concept words and differing number sets.

Public API:
    detect_contradiction(content_a, content_b, concept_a, concept_b) -> dict
"""

from __future__ import annotations

import re


def detect_contradiction(
    content_a: str,
    content_b: str,
    concept_a: str = "",
    concept_b: str = "",
) -> dict:
    """Detect if two facts about the same concept contain contradictory numbers.

    Simple heuristic: if two facts share a concept and contain different numbers
    for what appears to be the same measurement, flag as contradiction.

    Args:
        content_a: Content of first fact
        content_b: Content of second fact
        concept_a: Concept of first fact (optional, improves accuracy)
        concept_b: Concept of second fact (optional, improves accuracy)

    Returns:
        Dict with contradiction info if found:
            {"contradiction": True, "conflicting_values": "3, 5 vs 7, 9"}
        Empty dict if no contradiction detected.
    """
    # Only check facts with overlapping concepts
    concept_words_a = set(concept_a.lower().split()) if concept_a else set()
    concept_words_b = set(concept_b.lower().split()) if concept_b else set()

    if not concept_words_a or not concept_words_b:
        return {}

    # Need at least one meaningful concept word in common
    common = concept_words_a & concept_words_b
    # Remove very short words
    common = {w for w in common if len(w) > 2}
    if not common:
        return {}

    # Extract numbers from both facts
    nums_a = re.findall(r"\b\d+(?:\.\d+)?\b", content_a)
    nums_b = re.findall(r"\b\d+(?:\.\d+)?\b", content_b)

    if not nums_a or not nums_b:
        return {}

    # If the facts share concept words but have different number sets,
    # check for direct conflicts (same position in similar sentences)
    nums_a_set = set(nums_a)
    nums_b_set = set(nums_b)

    # If there are numbers unique to each fact about the same topic,
    # that is a potential contradiction
    unique_to_a = nums_a_set - nums_b_set
    unique_to_b = nums_b_set - nums_a_set

    if unique_to_a and unique_to_b:
        return {
            "contradiction": True,
            "conflicting_values": f"{', '.join(sorted(unique_to_a))} vs {', '.join(sorted(unique_to_b))}",
        }

    return {}


__all__ = ["detect_contradiction"]
