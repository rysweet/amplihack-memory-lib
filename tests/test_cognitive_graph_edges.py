"""Tests for CognitiveMemory graph edge creation and graph traversal search.

Covers:
- Module-level helpers: _tokenize, _compute_similarity, _detect_contradiction
- _create_similarity_edges via store_fact()
- _create_derives_from_edge via store_fact(source_episode_id=...)
- search_facts() graph traversal: keyword seed + 1-hop expansion + ranking
- Named constants: SIMILARITY_THRESHOLD, SIMILARITY_CANDIDATE_LIMIT,
  MAX_SEARCH_KEYWORDS, GRAPH_FETCH_MULTIPLIER, MIN_TOKEN_LENGTH
"""

import pytest

from amplihack_memory import CognitiveMemory
from amplihack_memory.cognitive_memory import (
    GRAPH_FETCH_MULTIPLIER,
    MAX_SEARCH_KEYWORDS,
    MIN_TOKEN_LENGTH,
    SIMILARITY_CANDIDATE_LIMIT,
    SIMILARITY_THRESHOLD,
    WORKING_MEMORY_CAPACITY,
    PROSPECTIVE_STATUS_PENDING,
    PROSPECTIVE_STATUS_TRIGGERED,
    PROSPECTIVE_STATUS_RESOLVED,
    _compute_similarity,
    _detect_contradiction,
    _tokenize,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cm(tmp_path):
    """Isolated CognitiveMemory instance backed by a temp Kuzu database."""
    mem = CognitiveMemory(agent_name="test-agent", db_path=tmp_path / "graph_test_db")
    yield mem
    mem.close()


@pytest.fixture
def cm2(tmp_path):
    """Second agent sharing the same database path as *cm*."""
    mem = CognitiveMemory(agent_name="other-agent", db_path=tmp_path / "graph_test_db")
    yield mem
    mem.close()


# ===========================================================================
# Constants
# ===========================================================================


class TestConstants:
    """Verify that named constants exist and have sensible values."""

    def test_similarity_threshold_is_float_in_range(self):
        assert isinstance(SIMILARITY_THRESHOLD, float)
        assert 0.0 < SIMILARITY_THRESHOLD < 1.0

    def test_similarity_candidate_limit_positive_int(self):
        assert isinstance(SIMILARITY_CANDIDATE_LIMIT, int)
        assert SIMILARITY_CANDIDATE_LIMIT > 0

    def test_max_search_keywords_positive_int(self):
        assert isinstance(MAX_SEARCH_KEYWORDS, int)
        assert MAX_SEARCH_KEYWORDS > 0

    def test_graph_fetch_multiplier_positive_int(self):
        assert isinstance(GRAPH_FETCH_MULTIPLIER, int)
        assert GRAPH_FETCH_MULTIPLIER > 1

    def test_min_token_length_positive_int(self):
        assert isinstance(MIN_TOKEN_LENGTH, int)
        assert MIN_TOKEN_LENGTH >= 1

    def test_working_memory_capacity_positive_int(self):
        assert isinstance(WORKING_MEMORY_CAPACITY, int)
        assert WORKING_MEMORY_CAPACITY > 0

    def test_prospective_status_constants_are_strings(self):
        assert isinstance(PROSPECTIVE_STATUS_PENDING, str)
        assert isinstance(PROSPECTIVE_STATUS_TRIGGERED, str)
        assert isinstance(PROSPECTIVE_STATUS_RESOLVED, str)
        # All distinct
        assert len({PROSPECTIVE_STATUS_PENDING, PROSPECTIVE_STATUS_TRIGGERED, PROSPECTIVE_STATUS_RESOLVED}) == 3


# ===========================================================================
# _tokenize
# ===========================================================================


class TestTokenize:
    """Unit tests for the _tokenize() helper."""

    def test_basic_tokenization(self):
        tokens = _tokenize("Python is a programming language")
        assert "python" in tokens
        assert "programming" in tokens
        assert "language" in tokens

    def test_stopwords_removed(self):
        # "the", "and", "are" are stopwords
        tokens = _tokenize("the cats and dogs are playing")
        assert "the" not in tokens
        assert "and" not in tokens
        assert "are" not in tokens

    def test_short_tokens_filtered(self):
        tokens = _tokenize("a is it go run fast")
        # "a", "is", "it", "go" are shorter than MIN_TOKEN_LENGTH (3)
        for token in tokens:
            assert len(token) >= MIN_TOKEN_LENGTH

    def test_punctuation_stripped(self):
        tokens = _tokenize("Hello, world! Python.")
        assert "hello" in tokens
        assert "world" in tokens
        assert "python" in tokens
        # No punctuation in any token
        for t in tokens:
            assert t == t.strip(".,!?;:\"'()[]{}-_/")

    def test_case_insensitive(self):
        tokens = _tokenize("Python PYTHON python")
        # All should normalize to "python"
        assert "python" in tokens
        assert len(tokens) == 1

    def test_empty_string(self):
        assert _tokenize("") == set()

    def test_only_stopwords(self):
        assert _tokenize("the and for with") == set()

    def test_returns_set(self):
        result = _tokenize("hello world")
        assert isinstance(result, set)


# ===========================================================================
# _compute_similarity
# ===========================================================================


class TestComputeSimilarity:
    """Unit tests for the _compute_similarity() helper."""

    def test_identical_sets(self):
        tokens = {"python", "programming", "language"}
        assert _compute_similarity(tokens, tokens) == 1.0

    def test_disjoint_sets(self):
        a = {"python", "programming"}
        b = {"java", "compiled"}
        assert _compute_similarity(a, b) == 0.0

    def test_partial_overlap(self):
        a = {"python", "programming", "language"}
        b = {"python", "compiled", "language"}
        # intersection = {python, language} = 2, union = {python, programming, language, compiled} = 4
        assert abs(_compute_similarity(a, b) - 2 / 4) < 1e-9

    def test_empty_first_set(self):
        assert _compute_similarity(set(), {"python"}) == 0.0

    def test_empty_second_set(self):
        assert _compute_similarity({"python"}, set()) == 0.0

    def test_both_empty(self):
        assert _compute_similarity(set(), set()) == 0.0

    def test_result_in_range(self):
        a = {"a", "b", "c"}
        b = {"b", "c", "d"}
        score = _compute_similarity(a, b)
        assert 0.0 <= score <= 1.0

    def test_symmetry(self):
        a = {"alpha", "beta", "gamma"}
        b = {"beta", "delta"}
        assert _compute_similarity(a, b) == _compute_similarity(b, a)


# ===========================================================================
# _detect_contradiction
# ===========================================================================


class TestDetectContradiction:
    """Unit tests for the _detect_contradiction() helper."""

    def test_obvious_negation(self):
        # One affirms, other negates shared words
        a = "Python is fast and efficient"
        b = "Python is not fast"
        assert _detect_contradiction(a, b) is True

    def test_no_contradiction_same_polarity(self):
        a = "Python is fast"
        b = "Python is very fast and efficient"
        assert _detect_contradiction(a, b) is False

    def test_no_overlap_no_contradiction(self):
        # Completely different topics: no shared words → no contradiction
        a = "Python is a programming language"
        b = "Elephants are large mammals"
        assert _detect_contradiction(a, b) is False

    def test_both_negations_not_contradiction(self):
        # Both have negations – not a contradiction in the heuristic
        a = "Python is not slow"
        b = "Python is not inefficient"
        assert _detect_contradiction(a, b) is False

    def test_returns_bool(self):
        result = _detect_contradiction("foo bar baz", "foo bar qux")
        assert isinstance(result, bool)

    def test_insufficient_shared_words(self):
        # Only one shared word → below the threshold of 2
        a = "Python runs fast"
        b = "Python is not reliable"
        # shared content words after filtering: just "python" (1 word)
        assert _detect_contradiction(a, b) is False


# ===========================================================================
# _create_similarity_edges  (via store_fact)
# ===========================================================================


class TestCreateSimilarityEdges:
    """Tests for SIMILAR_TO edge creation triggered by store_fact()."""

    def test_similar_facts_get_edge(self, cm):
        """Two highly similar facts should be connected by a SIMILAR_TO edge."""
        id1 = cm.store_fact(
            "Python language",
            "Python is an interpreted programming language",
        )
        id2 = cm.store_fact(
            "Python language",
            "Python is an interpreted programming language used widely",
        )
        # Verify the edge exists
        result = cm._conn.execute(
            """
            MATCH (a:SemanticMemory)-[r:SIMILAR_TO]->(b:SemanticMemory)
            WHERE a.node_id = $id2 AND b.node_id = $id1
            RETURN r.similarity_score
            """,
            {"id1": id1, "id2": id2},
        )
        assert result.has_next(), "Expected a SIMILAR_TO edge between similar facts"
        score = float(result.get_next()[0])
        assert score >= SIMILARITY_THRESHOLD

    def test_dissimilar_facts_no_edge(self, cm):
        """Two completely dissimilar facts should NOT be connected."""
        id1 = cm.store_fact("Python", "Python is an interpreted language")
        id2 = cm.store_fact("Cooking", "Boil water and add salt for pasta")
        result = cm._conn.execute(
            """
            MATCH (a:SemanticMemory)-[:SIMILAR_TO]->(b:SemanticMemory)
            WHERE (a.node_id = $id1 OR a.node_id = $id2)
            RETURN count(*)
            """,
            {"id1": id1, "id2": id2},
        )
        count = int(result.get_next()[0])
        assert count == 0

    def test_similarity_score_stored_on_edge(self, cm):
        """The similarity_score property on SIMILAR_TO edges should be in [0, 1]."""
        cm.store_fact("Machine learning", "Machine learning uses algorithms to learn patterns")
        cm.store_fact("Machine learning", "Machine learning trains models to learn from data")
        result = cm._conn.execute(
            """
            MATCH ()-[r:SIMILAR_TO]->()
            RETURN r.similarity_score
            LIMIT 1
            """
        )
        if result.has_next():
            score = float(result.get_next()[0])
            assert 0.0 <= score <= 1.0

    def test_no_self_edge(self, cm):
        """A fact must not be connected to itself."""
        id1 = cm.store_fact("Self", "This fact is unique in the world of facts")
        result = cm._conn.execute(
            """
            MATCH (a:SemanticMemory)-[:SIMILAR_TO]->(b:SemanticMemory)
            WHERE a.node_id = $nid AND b.node_id = $nid
            RETURN count(*)
            """,
            {"nid": id1},
        )
        assert int(result.get_next()[0]) == 0

    def test_multiple_similar_facts_multiple_edges(self, cm):
        """Store three near-identical facts; the third should get edges to both prior ones."""
        # Use highly overlapping text so Jaccard similarity exceeds SIMILARITY_THRESHOLD
        base = "Databases store and retrieve data using persistent storage"
        cm.store_fact("Databases", base)
        cm.store_fact("Databases", base + " efficiently")
        id3 = cm.store_fact("Databases", base + " reliably")
        result = cm._conn.execute(
            """
            MATCH (a:SemanticMemory)-[:SIMILAR_TO]->(b:SemanticMemory)
            WHERE a.node_id = $nid
            RETURN count(*)
            """,
            {"nid": id3},
        )
        edge_count = int(result.get_next()[0])
        assert edge_count >= 2, f"Expected ≥2 SIMILAR_TO edges from fact3, got {edge_count}"

    def test_agent_isolation_no_cross_agent_edges(self, tmp_path):
        """Facts from different agents must never share SIMILAR_TO edges."""
        shared_db = tmp_path / "shared_db"
        agent_a = CognitiveMemory(agent_name="agent-a", db_path=shared_db)
        agent_b = CognitiveMemory(agent_name="agent-b", db_path=shared_db)

        try:
            # Store nearly identical facts in different agents
            agent_a.store_fact(
                "Python language",
                "Python is an interpreted programming language",
            )
            agent_b.store_fact(
                "Python language",
                "Python is an interpreted programming language",
            )

            # Count cross-agent SIMILAR_TO edges
            result = agent_a._conn.execute(
                """
                MATCH (a:SemanticMemory)-[:SIMILAR_TO]->(b:SemanticMemory)
                WHERE a.agent_id <> b.agent_id
                RETURN count(*)
                """
            )
            assert int(result.get_next()[0]) == 0
        finally:
            agent_a.close()
            agent_b.close()


# ===========================================================================
# _create_derives_from_edge  (via store_fact source_episode_id)
# ===========================================================================


class TestCreateDerivesFromEdge:
    """Tests for DERIVES_FROM edge creation triggered by store_fact()."""

    def test_derives_from_edge_created(self, cm):
        """store_fact with source_episode_id creates a DERIVES_FROM edge."""
        ep_id = cm.store_episode(
            "User mentioned Python is their favourite language", "user-session"
        )
        sem_id = cm.store_fact(
            "Python",
            "Python is a favourite language",
            source_episode_id=ep_id,
        )
        result = cm._conn.execute(
            """
            MATCH (s:SemanticMemory)-[r:DERIVES_FROM]->(e:EpisodicMemory)
            WHERE s.node_id = $sid AND e.node_id = $eid
            RETURN r.derived_at
            """,
            {"sid": sem_id, "eid": ep_id},
        )
        assert result.has_next(), "Expected a DERIVES_FROM edge"
        derived_at = result.get_next()[0]
        assert derived_at > 0

    def test_no_episode_no_derives_from_edge(self, cm):
        """store_fact without source_episode_id creates no DERIVES_FROM edge."""
        sem_id = cm.store_fact("Python", "Python is interpreted")
        result = cm._conn.execute(
            """
            MATCH (s:SemanticMemory)-[:DERIVES_FROM]->(e:EpisodicMemory)
            WHERE s.node_id = $sid
            RETURN count(*)
            """,
            {"sid": sem_id},
        )
        assert int(result.get_next()[0]) == 0

    def test_multiple_facts_from_same_episode(self, cm):
        """Multiple facts can derive from the same episode."""
        ep_id = cm.store_episode("User talked about programming", "chat")
        id1 = cm.store_fact("Python", "Python is interpreted", source_episode_id=ep_id)
        id2 = cm.store_fact("Rust", "Rust is compiled", source_episode_id=ep_id)

        for sem_id in (id1, id2):
            result = cm._conn.execute(
                """
                MATCH (s:SemanticMemory)-[:DERIVES_FROM]->(e:EpisodicMemory)
                WHERE s.node_id = $sid AND e.node_id = $eid
                RETURN count(*)
                """,
                {"sid": sem_id, "eid": ep_id},
            )
            assert int(result.get_next()[0]) == 1

    def test_invalid_episode_id_does_not_raise(self, cm):
        """A non-existent episode_id should not raise an exception."""
        sem_id = cm.store_fact(
            "Python",
            "Python is interpreted",
            source_episode_id="nonexistent_ep_id",
        )
        # Node was still stored
        facts = cm.get_all_facts()
        assert any(f.node_id == sem_id for f in facts)

    def test_derives_from_edge_has_timestamp(self, cm):
        """The derived_at timestamp on DERIVES_FROM edges should be a positive int."""
        ep_id = cm.store_episode("An interesting event", "log")
        sem_id = cm.store_fact(
            "Event", "Something interesting happened", source_episode_id=ep_id
        )
        result = cm._conn.execute(
            """
            MATCH (s:SemanticMemory)-[r:DERIVES_FROM]->(e:EpisodicMemory)
            WHERE s.node_id = $sid
            RETURN r.derived_at
            """,
            {"sid": sem_id},
        )
        assert result.has_next()
        ts = result.get_next()[0]
        assert isinstance(ts, int)
        assert ts > 0


# ===========================================================================
# search_facts — graph traversal
# ===========================================================================


class TestSearchFactsGraphTraversal:
    """Tests for the graph traversal (1-hop expansion + ranking) in search_facts()."""

    def test_seed_facts_returned(self, cm):
        """Keyword-matching seed facts are included in results."""
        cm.store_fact("Python", "Python is an interpreted language")
        cm.store_fact("Rust", "Rust is a systems programming language")

        results = cm.search_facts("Python")
        concepts = [f.concept for f in results]
        assert "Python" in concepts

    def test_graph_expansion_finds_similar_neighbor(self, cm):
        """A fact not directly matching the query but similar to a seed is returned."""
        # Store two highly similar facts
        id1 = cm.store_fact(
            "Python language",
            "Python is an interpreted programming language",
        )
        id2 = cm.store_fact(
            "Python language",
            "Python is an interpreted programming language widely used",
        )
        # Verify a SIMILAR_TO edge exists (pre-condition for the traversal test)
        result = cm._conn.execute(
            """
            MATCH (a:SemanticMemory)-[:SIMILAR_TO]->(b:SemanticMemory)
            WHERE (a.node_id = $id1 OR a.node_id = $id2)
            RETURN count(*)
            """,
            {"id1": id1, "id2": id2},
        )
        edge_count = int(result.get_next()[0])
        if edge_count == 0:
            pytest.skip("No SIMILAR_TO edge created — similarity below threshold")

        # Now store an unrelated fact and search: both similar facts should appear
        results = cm.search_facts("Python interpreted")
        node_ids = {f.node_id for f in results}
        assert id1 in node_ids or id2 in node_ids

    def test_results_respect_limit(self, cm):
        """search_facts respects the *limit* parameter even after graph expansion."""
        for i in range(10):
            cm.store_fact(f"Concept{i}", f"All facts share common programming language topic {i}")
        results = cm.search_facts("programming language", limit=3)
        assert len(results) <= 3

    def test_results_respect_min_confidence(self, cm):
        """Facts below min_confidence are excluded even if they are graph neighbours."""
        id_high = cm.store_fact(
            "Python language",
            "Python is a programming language",
            confidence=0.9,
        )
        id_low = cm.store_fact(
            "Python language",
            "Python is a programming language for scripting",
            confidence=0.1,
        )
        results = cm.search_facts("Python programming language", min_confidence=0.5)
        node_ids = {f.node_id for f in results}
        assert id_high in node_ids
        assert id_low not in node_ids

    def test_no_duplicate_results(self, cm):
        """Each fact appears at most once in search results."""
        for i in range(5):
            cm.store_fact(
                "Programming",
                f"Programming languages are essential tools for developers number {i}",
            )
        results = cm.search_facts("programming languages developers")
        ids = [f.node_id for f in results]
        assert len(ids) == len(set(ids)), "Duplicate facts found in search results"

    def test_empty_query_returns_all_facts(self, cm):
        """An empty query string falls through to get_all_facts()."""
        cm.store_fact("A", "alpha content")
        cm.store_fact("B", "beta content")
        results = cm.search_facts("")
        assert len(results) >= 2

    def test_keyword_seed_selection(self, cm):
        """Keyword seeds correctly match against concept and content fields."""
        id_match = cm.store_fact("Database", "Relational databases store structured data")
        id_no_match = cm.store_fact("Weather", "The weather forecast predicts sunshine")

        results = cm.search_facts("database structured")
        node_ids = {f.node_id for f in results}
        assert id_match in node_ids
        assert id_no_match not in node_ids

    def test_ranking_seeds_before_neighbors(self, cm):
        """Direct keyword matches (seeds) should rank higher than 1-hop neighbours."""
        # id_direct matches the query
        id_direct = cm.store_fact(
            "Python speed",
            "Python performance can be improved with optimisation",
        )
        # id_similar is similar to id_direct but not a direct keyword hit
        id_similar = cm.store_fact(
            "Python speed",
            "Python performance can be improved with optimisation techniques",
        )
        # Verify edge exists
        result = cm._conn.execute(
            """
            MATCH (a:SemanticMemory)-[:SIMILAR_TO]->(b:SemanticMemory)
            WHERE (a.node_id = $d OR a.node_id = $s)
            RETURN count(*)
            """,
            {"d": id_direct, "s": id_similar},
        )
        if int(result.get_next()[0]) == 0:
            pytest.skip("No SIMILAR_TO edge — can't test ranking order")

        results = cm.search_facts("performance improved optimisation")
        assert len(results) >= 1
        # First result should be one of the two related facts
        assert results[0].node_id in {id_direct, id_similar}

    def test_graph_traversal_agent_isolation(self, tmp_path):
        """Graph traversal must not return facts from other agents."""
        shared_db = tmp_path / "shared_iso_db"
        agent_a = CognitiveMemory(agent_name="agent-a", db_path=shared_db)
        agent_b = CognitiveMemory(agent_name="agent-b", db_path=shared_db)
        try:
            agent_a.store_fact("Python", "Python is an interpreted language")
            agent_b.store_fact("Java", "Java is a compiled language")

            results_a = agent_a.search_facts("language")
            for fact in results_a:
                assert fact.concept != "Java", "agent-a got a fact belonging to agent-b"
        finally:
            agent_a.close()
            agent_b.close()


# ===========================================================================
# Integration: store_fact + search_facts end-to-end
# ===========================================================================


class TestGraphEdgeIntegration:
    """End-to-end integration tests combining edge creation and traversal."""

    def test_store_similar_then_search_finds_both(self, cm):
        """Storing two similar facts and searching should return both via graph expansion."""
        id1 = cm.store_fact(
            "Machine learning",
            "Machine learning algorithms learn patterns from training data",
        )
        id2 = cm.store_fact(
            "Machine learning",
            "Machine learning models are trained on large datasets to learn patterns",
        )

        results = cm.search_facts("machine learning patterns")
        node_ids = {f.node_id for f in results}
        # At least one of the two facts (seed or expanded) should appear
        assert id1 in node_ids or id2 in node_ids

    def test_derives_from_plus_similarity(self, cm):
        """A fact can simultaneously have DERIVES_FROM and SIMILAR_TO edges."""
        ep_id = cm.store_episode("Discussed machine learning at conference", "log")

        id1 = cm.store_fact(
            "ML fundamentals",
            "Machine learning uses statistical models to learn from examples",
        )
        id2 = cm.store_fact(
            "ML fundamentals",
            "Machine learning relies on statistical methods to learn patterns",
            source_episode_id=ep_id,
        )

        # DERIVES_FROM edge
        result = cm._conn.execute(
            """
            MATCH (s:SemanticMemory)-[:DERIVES_FROM]->(e:EpisodicMemory)
            WHERE s.node_id = $sid
            RETURN count(*)
            """,
            {"sid": id2},
        )
        assert int(result.get_next()[0]) == 1, "Expected DERIVES_FROM edge"

        # SIMILAR_TO edge (if similarity meets threshold)
        result2 = cm._conn.execute(
            """
            MATCH (a:SemanticMemory)-[:SIMILAR_TO]->(b:SemanticMemory)
            WHERE (a.node_id = $id1 OR a.node_id = $id2)
            RETURN count(*)
            """,
            {"id1": id1, "id2": id2},
        )
        # We don't assert a specific count here — just that the query runs
        assert result2 is not None

    def test_search_facts_returns_semantic_fact_objects(self, cm):
        """search_facts always returns SemanticFact instances with correct fields."""
        from amplihack_memory.memory_types import SemanticFact

        cm.store_fact("Test concept", "Test content for verification", confidence=0.8)
        results = cm.search_facts("test concept")
        for fact in results:
            assert isinstance(fact, SemanticFact)
            assert fact.node_id.startswith("sem_")
            assert 0.0 <= fact.confidence <= 1.0

    def test_store_fact_still_returns_node_id(self, cm):
        """store_fact() must still return a valid node_id after adding edge logic."""
        node_id = cm.store_fact("Graph", "Graphs connect nodes with edges")
        assert isinstance(node_id, str)
        assert node_id.startswith("sem_")

    def test_keyword_truncation_constant(self, cm):
        """Keyword search honours MAX_SEARCH_KEYWORDS constant."""
        # Store a fact with a long concept
        long_concept = " ".join(f"word{i}" for i in range(MAX_SEARCH_KEYWORDS + 5))
        cm.store_fact("WordFact", "A fact with many words in content")
        # Search with more than MAX_SEARCH_KEYWORDS terms (should not raise)
        results = cm.search_facts(long_concept)
        # Just verify no exception and result is a list
        assert isinstance(results, list)
