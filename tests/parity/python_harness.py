#!/usr/bin/env python3
"""A/B parity test harness — Python side.

Exercises every public API and writes structured JSON to stdout
so the Rust harness output can be compared.
"""

import json
import sys
import traceback
from datetime import datetime

results = []


def record(test_name: str, result, success: bool = True):
    results.append({"test": test_name, "result": result, "success": success})


def safe_run(test_name: str, fn):
    try:
        fn()
    except Exception as exc:
        record(test_name, {"error": str(exc), "traceback": traceback.format_exc()}, success=False)


# ── 1. Experience ─────────────────────────────────────────────────────
def test_experience():
    from amplihack_memory import Experience, ExperienceType

    exp = Experience(
        experience_type=ExperienceType.SUCCESS,
        context="Deployed model to production",
        outcome="Model accuracy improved by 15%",
        confidence=0.85,
        tags=["deployment", "ml"],
    )
    d = exp.to_dict()
    record("experience_create", {
        "experience_type": d["experience_type"],
        "context": d["context"],
        "outcome": d["outcome"],
        "confidence": d["confidence"],
        "has_id": bool(d.get("experience_id")),
        "id_prefix": d["experience_id"][:4] if d.get("experience_id") else "",
        "has_timestamp": bool(d.get("timestamp")),
        "tags": d.get("tags", []),
        "metadata": d.get("metadata", {}),
    })

    # Validation: empty context should fail
    try:
        Experience(
            experience_type=ExperienceType.SUCCESS,
            context="",
            outcome="some outcome",
            confidence=0.5,
        )
        record("experience_empty_context_rejected", False)
    except (ValueError, Exception):
        record("experience_empty_context_rejected", True)

    # Validation: confidence out of range
    try:
        Experience(
            experience_type=ExperienceType.SUCCESS,
            context="ctx",
            outcome="outcome",
            confidence=1.5,
        )
        record("experience_confidence_range", False)
    except (ValueError, Exception):
        record("experience_confidence_range", True)

safe_run("experience", test_experience)


# ── 2. Similarity ────────────────────────────────────────────────────
def test_similarity():
    from amplihack_memory import compute_similarity, compute_word_similarity, compute_tag_similarity

    # Word similarity — same text
    ws_identical = compute_word_similarity("hello world test", "hello world test")
    record("word_similarity_identical", round(ws_identical, 4))

    # Word similarity — no overlap
    ws_none = compute_word_similarity("alpha beta gamma", "delta epsilon zeta")
    record("word_similarity_no_overlap", round(ws_none, 4))

    # Word similarity — partial
    ws_partial = compute_word_similarity("machine learning AI", "artificial intelligence ML")
    record("word_similarity_partial", round(ws_partial, 4))

    # Tag similarity — Jaccard on known sets
    ts = compute_tag_similarity(["rust", "python"], ["python", "go"])
    record("tag_similarity", round(ts, 4))

    # Composite similarity — node dicts
    node_a = {"content": "Rust programming language memory safety",
              "concept": "rust", "tags": ["language", "systems"]}
    node_b = {"content": "Rust programming compiler optimization",
              "concept": "rust", "tags": ["language", "compiler"]}
    cs = compute_similarity(node_a, node_b)
    record("compute_similarity_partial", round(cs, 4))

    # Composite similarity — identical nodes
    node_same = {"content": "Rust programming language is fast and safe",
                 "concept": "rust", "tags": ["language", "systems"]}
    cs_id = compute_similarity(node_same, dict(node_same))
    record("compute_similarity_identical", round(cs_id, 4))

safe_run("similarity", test_similarity)


# ── 3. Entity extraction ─────────────────────────────────────────────
def test_entity_extraction():
    from amplihack_memory import extract_entity_name

    e1 = extract_entity_name("John Smith works at Microsoft in Seattle", "")
    record("entity_extraction_multi_word", e1)

    e2 = extract_entity_name("some content", "Sarah Chen")
    record("entity_extraction_concept_first", e2)

    e3 = extract_entity_name("", "")
    record("entity_extraction_empty", e3)

    e4 = extract_entity_name("all lowercase words here", "")
    record("entity_extraction_no_names", e4)

safe_run("entity_extraction", test_entity_extraction)


# ── 4. Contradiction detection ───────────────────────────────────────
def test_contradiction():
    from amplihack_memory import detect_contradiction

    # Should detect contradiction
    r1 = detect_contradiction(
        "The temperature is 72 degrees",
        "The temperature is 45 degrees",
        "temperature",
        "temperature",
    )
    record("contradiction_detected", {
        "has_contradiction": bool(r1 and r1.get("contradiction")),
        "conflicting_values": r1.get("conflicting_values", "") if r1 else "",
    })

    # No contradiction — same numbers
    r2 = detect_contradiction(
        "Team has 5 members",
        "Team has 5 members",
        "team size",
        "team size",
    )
    record("contradiction_same_numbers", {
        "has_contradiction": bool(r2 and r2.get("contradiction")),
    })

    # No contradiction — different concepts
    r3 = detect_contradiction(
        "Has 5 members",
        "Has 8 items",
        "team",
        "inventory",
    )
    record("contradiction_different_concepts", {
        "has_contradiction": bool(r3 and r3.get("contradiction")),
    })

    # No numbers
    r4 = detect_contradiction(
        "The team is large",
        "The team is small",
        "team size",
        "team size",
    )
    record("contradiction_no_numbers", {
        "has_contradiction": bool(r4 and r4.get("contradiction")),
    })

safe_run("contradiction", test_contradiction)


# ── 5. Security — credential scrubbing ───────────────────────────────
def test_credential_scrubbing():
    from amplihack_memory import CredentialScrubber

    scrubber = CredentialScrubber()

    text_with_creds = (
        "AWS key AKIAIOSFODNN7EXAMPLE found. "
        "GitHub token ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef12. "
        "password=mysecret123"
    )

    contains = scrubber.contains_credentials(text_with_creds)
    record("credentials_detected", contains)

    scrubbed, was_modified = scrubber._scrub_text(text_with_creds)
    record("credentials_scrubbed", {
        "was_modified": was_modified,
        "contains_aws_key": "AKIAIOSFODNN7EXAMPLE" in scrubbed,
        "contains_github_token": "ghp_ABCDEF" in scrubbed,
        "contains_password": "mysecret123" in scrubbed,
        "contains_redacted": "[REDACTED]" in scrubbed,
    })

    clean_text = "This text has no credentials at all"
    clean_contains = scrubber.contains_credentials(clean_text)
    record("credentials_clean_text", not clean_contains)

safe_run("credential_scrubbing", test_credential_scrubbing)


# ── 6. Security — query validation ──────────────────────────────────
def test_query_validation():
    from amplihack_memory import QueryValidator

    validator = QueryValidator()

    safe = validator.is_safe_query("SELECT * FROM experiences")
    record("query_safe_select", safe)

    unsafe = validator.is_safe_query("DROP TABLE experiences")
    record("query_unsafe_drop", not unsafe)  # True means it correctly rejected

    unsafe2 = validator.is_safe_query("DELETE FROM experiences WHERE id = 1")
    record("query_unsafe_delete", not unsafe2)

    cost = validator.estimate_cost("SELECT * FROM experiences")
    record("query_cost_simple_select", cost)

    cost_join = validator.estimate_cost(
        "SELECT * FROM experiences JOIN tags ON experiences.id = tags.exp_id LIMIT 10"
    )
    record("query_cost_with_join", cost_join)

safe_run("query_validation", test_query_validation)


# ── 7. CognitiveMemory ──────────────────────────────────────────────
def test_cognitive_memory():
    import tempfile, os
    from amplihack_memory import CognitiveMemory

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_db")
        cm = CognitiveMemory(agent_name="test-agent", db_path=db_path)

        # Store a fact
        fact_id = cm.store_fact(
            content="Python is a programming language",
            concept="python",
            confidence=0.9,
            tags=["language", "programming"],
        )
        record("cognitive_store_fact", {
            "has_id": bool(fact_id),
            "id_type": type(fact_id).__name__,
        })

        # Store an episode
        ep_id = cm.store_episode(
            content="Deployed the new ML model successfully",
            source_label="deployment-session",
        )
        record("cognitive_store_episode", {
            "has_id": bool(ep_id),
            "id_type": type(ep_id).__name__,
        })

        # Push working memory
        wm_id = cm.push_working(
            slot_type="goal",
            content="Complete the deployment pipeline",
            task_id="task-001",
            relevance=0.95,
        )
        record("cognitive_push_working", {
            "has_id": bool(wm_id),
            "id_type": type(wm_id).__name__,
        })

        # Get working memory
        wm_slots = cm.get_working("task-001")
        record("cognitive_get_working", {
            "count": len(wm_slots),
            "first_content": wm_slots[0].content if wm_slots else "",
        })

        # Get stats
        stats = cm.get_statistics()
        record("cognitive_stats", {
            "has_semantic": "semantic" in stats,
            "has_episodic": "episodic" in stats,
            "has_working": "working" in stats,
            "semantic_count": stats.get("semantic", 0),
            "episodic_count": stats.get("episodic", 0),
            "working_count": stats.get("working", 0),
        })

safe_run("cognitive_memory", test_cognitive_memory)


# ── 8. Pattern recognition ──────────────────────────────────────────
def test_pattern_recognition():
    from amplihack_memory.pattern_recognition import (
        PatternDetector,
        extract_pattern_key,
        calculate_pattern_confidence,
    )

    detector = PatternDetector(threshold=3, min_confidence=0.5)

    # Add discoveries below threshold
    for _ in range(2):
        detector.add_discovery({"type": "test_type"})
    record("pattern_below_threshold", not detector.is_pattern_recognized("test_type"))

    # Add one more to reach threshold
    detector.add_discovery({"type": "test_type"})
    record("pattern_at_threshold", detector.is_pattern_recognized("test_type"))
    record("pattern_occurrence_count", detector.get_occurrence_count("test_type"))

    # Pattern key extraction
    key1 = extract_pattern_key({"type": "test_type"})
    record("pattern_key_simple", key1)

    key2 = extract_pattern_key({"type": "link", "link_type": "similar"})
    record("pattern_key_with_link", key2)

    # Confidence calculation
    conf = calculate_pattern_confidence(3, 3)
    record("pattern_confidence_at_3", round(conf, 4))

    conf5 = calculate_pattern_confidence(5, 3)
    record("pattern_confidence_at_5", round(conf5, 4))

safe_run("pattern_recognition", test_pattern_recognition)


# ── Output ───────────────────────────────────────────────────────────
print(json.dumps(results, indent=2, default=str))
