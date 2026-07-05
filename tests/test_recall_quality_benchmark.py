"""Durable recall-quality regression benchmark (precision@k + MRR).

This is **internal test infrastructure**, not a public API. It guards the
"recall quality" axis of the memory library: given a fixed, hand-labelled
fixture corpus of agent experiences and a set of natural-language queries with
known-relevant results, it measures how well the existing recall/ranking path
(:func:`amplihack_memory.semantic_search.retrieve_relevant_experiences`) surfaces
the relevant experiences at the top of the ranking.

Two standard information-retrieval metrics are computed over the fixed corpus:

* **precision@k** — of the top ``k`` ranked results for a query, what fraction
  are relevant (averaged over all queries).
* **MRR** (mean reciprocal rank) — the average of ``1 / rank`` of the first
  relevant result for each query.

The benchmark is fully deterministic: the corpus uses explicit, stable
``experience_id`` values and a single shared timestamp (so the recency factor
is uniform and never affects relative ordering), and the ranking path itself is
pure Python with no ML dependencies. Because the metrics depend only on the
*relative* ordering of results, the recorded baselines below are byte-stable
across runs and platforms.

The regression tests assert the metrics never drop below the recorded baselines,
so any future change that degrades recall ranking fails CI (the ``python-tests``
job runs this file via ``pytest tests/``).

Reusable entry points (importable by other harnesses):
    RECALL_BENCH_CORPUS      - the fixed list[Experience]
    RECALL_BENCH_QUERIES     - dict[query -> set[relevant experience_id]]
    precision_at_k(...)      - single-query precision@k
    reciprocal_rank(...)     - single-query reciprocal rank
    evaluate_recall(k)       - (mean_precision_at_k, mrr) over the corpus

Run standalone to print a report:
    uv run python tests/test_recall_quality_benchmark.py
"""

from __future__ import annotations

from datetime import datetime

from amplihack_memory.experience import Experience, ExperienceType
from amplihack_memory.semantic_search import retrieve_relevant_experiences

# Single shared timestamp: makes the recency factor uniform across the corpus so
# it never influences the *relative* ranking the benchmark measures.
_BENCH_TS = datetime(2026, 1, 1, 12, 0, 0)

_P = ExperienceType.PATTERN
_S = ExperienceType.SUCCESS
_F = ExperienceType.FAILURE
_I = ExperienceType.INSIGHT


def _exp(exp_id: str, exp_type: ExperienceType, context: str) -> Experience:
    """Build a fixture experience with a stable id and shared timestamp."""
    return Experience(
        experience_type=exp_type,
        context=context,
        outcome="outcome recorded",
        confidence=0.8,
        timestamp=_BENCH_TS,
        experience_id=exp_id,
    )


# --- Fixed fixture corpus ---------------------------------------------------
#
# A small, realistic set of stored agent experiences spanning several topics.
# Items e17-e20 are deliberately *topically unrelated* notes that nonetheless
# share generic question-scaffolding words ("how do we", "how to", "reduce",
# "improve") with the queries. They act as adversarial-but-realistic distractors
# that a naive (stop-word-blind) tokenizer would rank too highly.
RECALL_BENCH_CORPUS: list[Experience] = [
    _exp("e01", _S, "The SQL injection vulnerability was found in the login form."),
    _exp("e02", _P, "Parameterized queries prevent SQL injection in user input."),
    _exp("e03", _S, "Cross-site scripting was detected in the comment field."),
    _exp("e04", _S, "The Python documentation was missing runnable examples."),
    _exp("e05", _P, "Good docstrings raise Python documentation quality."),
    _exp("e06", _S, "The JavaScript code review completed with no issues."),
    _exp("e07", _S, "Database read latency dropped after we added an index."),
    _exp("e08", _P, "Missing indexes cause slow database scans under load."),
    _exp("e09", _I, "Refactoring into small functions improves maintainability."),
    _exp("e10", _S, "Unit test coverage was raised from 60 percent to 90 percent."),
    _exp("e11", _P, "Flaky tests are usually caused by shared mutable state."),
    _exp("e12", _S, "The memory leak was traced to an unclosed file handle."),
    _exp("e13", _S, "Response caching cut the API latency roughly in half."),
    _exp("e14", _F, "The deployment failed due to a missing environment variable."),
    _exp("e15", _I, "Clear error messages reduce debugging time for engineers."),
    _exp("e16", _S, "The CSV parser crashed on rows with embedded commas."),
    _exp("e17", _S, "How do we stop the staging server before the nightly job?"),
    _exp("e18", _S, "Notes on how to run the release and how to prevent downtime."),
    _exp("e19", _S, "A quick guide on how to improve the onboarding docs for users."),
    _exp("e20", _S, "We need to reduce the noise in the daily standup meeting."),
]

# --- Query -> known-relevant experience ids ---------------------------------
RECALL_BENCH_QUERIES: dict[str, set[str]] = {
    "How do we prevent SQL injection?": {"e01", "e02"},
    "How to improve Python documentation?": {"e04", "e05"},
    "Why are the database queries so slow?": {"e07", "e08"},
    "How do we reduce flaky tests?": {"e10", "e11"},
    "Why did the deployment fail?": {"e14"},
    "How can we reduce API latency?": {"e13"},
    "How to trace a memory leak?": {"e12"},
    "How do we run a code review?": {"e06"},
}

# --- Recorded baselines (captured on this fixed corpus) ---------------------
#
# Captured from the current ranking path. The stop-word/punctuation-blind
# tokenizer previously scored precision@3 = 0.2500 and MRR = 0.4938 on this
# corpus; the content-aware tokenizer fix raised them to the values below.
# CI asserts the live metrics never regress below these.
BASELINE_PRECISION_AT_3 = 0.4583333333333333  # 11/24
BASELINE_MRR = 1.0
BENCH_K = 3

# Tiny tolerance to absorb floating-point representation differences only.
_EPS = 1e-9


def precision_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of the top-``k`` ranked ids that are relevant.

    Args:
        ranked_ids: Ranked result ids, best first.
        relevant_ids: The set of ids that are relevant for the query.
        k: Cutoff rank (must be > 0).

    Returns:
        precision@k in [0.0, 1.0].
    """
    if k <= 0:
        raise ValueError("k must be > 0")
    top_k = ranked_ids[:k]
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / k


def reciprocal_rank(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    """Reciprocal of the rank (1-based) of the first relevant id, else 0.0."""
    for position, rid in enumerate(ranked_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / position
    return 0.0


def _rank_query(query: str) -> list[str]:
    """Return the full ranked list of experience ids for a query."""
    results = retrieve_relevant_experiences(
        RECALL_BENCH_CORPUS, query, top_k=len(RECALL_BENCH_CORPUS)
    )
    return [exp.experience_id for exp in results]


def evaluate_recall(k: int = BENCH_K) -> tuple[float, float]:
    """Evaluate the recall path over the fixed corpus.

    Args:
        k: precision cutoff.

    Returns:
        (mean_precision_at_k, mean_reciprocal_rank) averaged over all queries.
    """
    precisions: list[float] = []
    rrs: list[float] = []
    for query, relevant in RECALL_BENCH_QUERIES.items():
        ranked = _rank_query(query)
        precisions.append(precision_at_k(ranked, relevant, k))
        rrs.append(reciprocal_rank(ranked, relevant))
    n = len(RECALL_BENCH_QUERIES)
    return sum(precisions) / n, sum(rrs) / n


class TestRecallQualityBenchmark:
    """Regression guard: recall metrics must not drop below recorded baselines."""

    def test_corpus_and_labels_are_wellformed(self):
        ids = [exp.experience_id for exp in RECALL_BENCH_CORPUS]
        assert len(ids) == len(set(ids)), "corpus experience ids must be unique"
        known = set(ids)
        for query, relevant in RECALL_BENCH_QUERIES.items():
            assert relevant, f"query has no labelled relevant ids: {query!r}"
            missing = relevant - known
            assert not missing, f"query {query!r} references unknown ids: {missing}"

    def test_precision_at_3_meets_baseline(self):
        precision, _ = evaluate_recall(BENCH_K)
        assert precision >= BASELINE_PRECISION_AT_3 - _EPS, (
            f"precision@{BENCH_K} regressed: {precision:.6f} "
            f"< baseline {BASELINE_PRECISION_AT_3:.6f}"
        )

    def test_mrr_meets_baseline(self):
        _, mrr = evaluate_recall(BENCH_K)
        assert mrr >= BASELINE_MRR - _EPS, (
            f"MRR regressed: {mrr:.6f} < baseline {BASELINE_MRR:.6f}"
        )

    def test_metric_helpers_are_correct(self):
        # Sanity checks for the metric primitives themselves.
        assert precision_at_k(["a", "b", "c"], {"a", "c"}, 3) == 2 / 3
        assert precision_at_k(["x", "y"], {"z"}, 2) == 0.0
        assert reciprocal_rank(["x", "a", "b"], {"a"}) == 0.5
        assert reciprocal_rank(["x", "y"], {"z"}) == 0.0


def _report() -> None:
    precision, mrr = evaluate_recall(BENCH_K)
    print("Recall-quality benchmark (fixed corpus)")
    print(f"  queries              : {len(RECALL_BENCH_QUERIES)}")
    print(f"  corpus size          : {len(RECALL_BENCH_CORPUS)}")
    print(f"  precision@{BENCH_K}          : {precision:.6f} "
          f"(baseline {BASELINE_PRECISION_AT_3:.6f})")
    print(f"  MRR                  : {mrr:.6f} (baseline {BASELINE_MRR:.6f})")


if __name__ == "__main__":
    _report()
