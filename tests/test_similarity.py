"""Tests for similarity module."""

from amplihack_memory.similarity import (
    _tokenize,
    compute_similarity,
    compute_tag_similarity,
    compute_word_similarity,
    rerank_facts_by_query,
)


class TestTokenize:
    """Tests for _tokenize function."""

    def test_empty_string(self):
        assert _tokenize("") == set()

    def test_removes_stop_words(self):
        tokens = _tokenize("the quick brown fox is very fast")
        assert "the" not in tokens
        assert "is" not in tokens
        assert "very" not in tokens

    def test_removes_short_tokens(self):
        tokens = _tokenize("a I do it")
        # All short or stop words, should be empty or near empty
        assert len(tokens) == 0

    def test_strips_punctuation(self):
        tokens = _tokenize("hello, world! test.")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens

    def test_lowercases(self):
        tokens = _tokenize("Hello WORLD Test")
        assert "hello" in tokens
        assert "world" in tokens


class TestComputeWordSimilarity:
    """Tests for compute_word_similarity."""

    def test_identical_texts(self):
        sim = compute_word_similarity(
            "plants use photosynthesis for energy",
            "plants use photosynthesis for energy",
        )
        assert sim == 1.0

    def test_completely_different(self):
        sim = compute_word_similarity(
            "quantum physics experiments",
            "chocolate cake recipes baking",
        )
        assert sim == 0.0

    def test_partial_overlap(self):
        sim = compute_word_similarity(
            "plants use photosynthesis",
            "photosynthesis provides energy plants",
        )
        assert 0.0 < sim < 1.0

    def test_empty_text(self):
        assert compute_word_similarity("", "hello world") == 0.0
        assert compute_word_similarity("hello", "") == 0.0
        assert compute_word_similarity("", "") == 0.0


class TestComputeTagSimilarity:
    """Tests for compute_tag_similarity."""

    def test_identical_tags(self):
        sim = compute_tag_similarity(["biology", "science"], ["biology", "science"])
        assert sim == 1.0

    def test_no_overlap(self):
        sim = compute_tag_similarity(["biology", "science"], ["history", "art"])
        assert sim == 0.0

    def test_partial_overlap(self):
        sim = compute_tag_similarity(["biology", "science"], ["biology", "chemistry"])
        assert 0.0 < sim < 1.0

    def test_empty_tags(self):
        assert compute_tag_similarity([], ["biology"]) == 0.0
        assert compute_tag_similarity(["biology"], []) == 0.0
        assert compute_tag_similarity([], []) == 0.0

    def test_case_insensitive(self):
        sim = compute_tag_similarity(["Biology"], ["biology"])
        assert sim == 1.0


class TestComputeSimilarity:
    """Tests for compute_similarity (composite)."""

    def test_identical_nodes(self):
        node = {"content": "photosynthesis in plants", "tags": ["biology"], "concept": "biology"}
        sim = compute_similarity(node, node)
        assert sim == 1.0

    def test_different_nodes(self):
        node_a = {"content": "quantum physics theory", "tags": ["physics"], "concept": "physics"}
        node_b = {"content": "chocolate cake recipe", "tags": ["cooking"], "concept": "cooking"}
        sim = compute_similarity(node_a, node_b)
        assert sim < 0.1

    def test_weighted_components(self):
        """Verify weighting: 0.5 content + 0.2 tags + 0.3 concept."""
        node_a = {"content": "photosynthesis", "tags": ["biology"], "concept": "biology"}
        node_b = {"content": "completely different", "tags": ["biology"], "concept": "biology"}
        sim = compute_similarity(node_a, node_b)
        # Tags match (0.2 * 1.0) + concept match (0.3 * 1.0) = 0.5
        # Content differs so word_sim ~ 0
        assert sim >= 0.4  # tags + concept contribute


class TestRerankFactsByQuery:
    """Tests for rerank_facts_by_query."""

    def test_empty_facts(self):
        assert rerank_facts_by_query([], "test") == []

    def test_empty_query(self):
        facts = [{"outcome": "test"}]
        assert rerank_facts_by_query(facts, "") == facts

    def test_relevant_first(self):
        facts = [
            {"context": "unrelated topic cooking", "outcome": "delicious food"},
            {"context": "photosynthesis in plants", "outcome": "energy production"},
        ]
        reranked = rerank_facts_by_query(facts, "photosynthesis plants")
        assert reranked[0]["outcome"] == "energy production"

    def test_top_k(self):
        facts = [
            {"context": "fact1", "outcome": "a"},
            {"context": "fact2", "outcome": "b"},
            {"context": "fact3", "outcome": "c"},
        ]
        result = rerank_facts_by_query(facts, "fact1", top_k=1)
        assert len(result) == 1

    def test_temporal_boost(self):
        """Facts with temporal metadata get boosted for temporal queries."""
        facts = [
            {"context": "gold medals count", "outcome": "won medals", "metadata": {}},
            {
                "context": "gold medals count",
                "outcome": "won medals",
                "metadata": {"temporal_index": 2, "source_date": "2024-01-01"},
            },
        ]
        reranked = rerank_facts_by_query(facts, "how did medals change over time")
        # Temporal fact should be first due to boost
        assert reranked[0]["metadata"].get("temporal_index") == 2

    def test_preserves_all_facts_without_top_k(self):
        facts = [
            {"context": "a", "outcome": "1"},
            {"context": "b", "outcome": "2"},
            {"context": "c", "outcome": "3"},
        ]
        result = rerank_facts_by_query(facts, "test query")
        assert len(result) == 3
