"""Entity name extraction from text and concept fields.

Extracts proper nouns (capitalized multi-word names) from text content,
handling apostrophe names (O'Brien), hyphenated names (Al-Hassan), and
standard multi-word names (Sarah Chen).

Public API:
    extract_entity_name(content, concept) -> str
"""

from __future__ import annotations

import re


def extract_entity_name(content: str, concept: str = "") -> str:
    """Extract the primary entity name from content or concept.

    Uses simple heuristics to find proper nouns (capitalized multi-word names)
    in the concept field first, then the content. Returns lowercase for
    consistent indexing.

    Args:
        content: Fact content text
        concept: Concept/topic label (checked first, usually more specific)

    Returns:
        Lowercased entity name, or empty string if none found.
    """
    # Try concept field first (usually more specific)
    for text in [concept, content]:
        if not text:
            continue
        # Find capitalized multi-word names (e.g., "Sarah Chen", "O'Brien Smith", "Al-Hassan")
        matches = re.findall(
            r"\b("
            r"[A-Z][a-z]*(?:['\u2019\-][A-Z]?[a-z]+)+(?:\s+(?:[A-Z][a-z]+(?:['\u2019\-][A-Z]?[a-z]+)?))*"
            r"|"
            r"[A-Z][a-z]+(?:\s+(?:[A-Z][a-z]+(?:['\u2019\-][A-Z]?[a-z]+)?))+)"
            r"\b",
            text,
        )
        if matches:
            # Return the longest match (most likely to be a full name)
            best = max(matches, key=len)
            return best.lower()

        # Single capitalized word that isn't at start of sentence
        # and isn't a common word
        words = text.split()
        for i, word in enumerate(words):
            if i > 0 and word[0:1].isupper() and len(word) > 2:
                cleaned = word.strip(".,;:!?()[]{}\"'")
                if cleaned and cleaned[0].isupper():
                    return cleaned.lower()

    return ""


__all__ = ["extract_entity_name"]
