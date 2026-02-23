"""Tests for the six-type cognitive memory system."""

import time

import pytest

from amplihack_memory import CognitiveMemory
from amplihack_memory.memory_types import (
    ConsolidatedEpisode,
    EpisodicMemory,
    MemoryCategory,
    ProceduralMemory,
    ProspectiveMemory,
    SemanticFact,
    SensoryItem,
    WorkingMemorySlot,
)


@pytest.fixture
def cm(tmp_path):
    """Create a CognitiveMemory instance with an isolated temp database."""
    db = tmp_path / "test_cognitive_db"
    mem = CognitiveMemory(agent_name="test-agent", db_path=db)
    yield mem
    mem.close()


@pytest.fixture
def cm_other(tmp_path):
    """Second CognitiveMemory instance for agent-isolation tests."""
    db = tmp_path / "test_cognitive_db"
    mem = CognitiveMemory(agent_name="other-agent", db_path=db)
    yield mem
    mem.close()


# ==================================================================
# Initialisation
# ==================================================================


class TestInit:
    def test_empty_agent_name_raises(self, tmp_path):
        with pytest.raises(ValueError, match="agent_name"):
            CognitiveMemory(agent_name="", db_path=tmp_path / "db")

    def test_whitespace_agent_name_raises(self, tmp_path):
        with pytest.raises(ValueError, match="agent_name"):
            CognitiveMemory(agent_name="   ", db_path=tmp_path / "db")

    def test_creates_db_directory(self, tmp_path):
        db_path = tmp_path / "new_dir" / "db"
        cm = CognitiveMemory(agent_name="a", db_path=db_path)
        assert db_path.exists()
        cm.close()


# ==================================================================
# SENSORY MEMORY
# ==================================================================


class TestSensoryMemory:
    def test_record_and_retrieve(self, cm):
        nid = cm.record_sensory("text", "Hello world")
        assert nid.startswith("sen_")

        items = cm.get_recent_sensory(limit=5)
        assert len(items) == 1
        assert items[0].node_id == nid
        assert items[0].modality == "text"
        assert items[0].raw_data == "Hello world"

    def test_ordering(self, cm):
        cm.record_sensory("a", "first")
        cm.record_sensory("b", "second")
        cm.record_sensory("c", "third")

        items = cm.get_recent_sensory(limit=10)
        assert len(items) == 3
        # Most recent first
        assert items[0].raw_data == "third"
        assert items[2].raw_data == "first"

    def test_expired_not_returned(self, cm):
        cm.record_sensory("text", "ephemeral", ttl_seconds=0)
        # TTL=0 means expires_at == created_at; already expired
        time.sleep(0.1)
        items = cm.get_recent_sensory()
        assert len(items) == 0

    def test_prune_expired(self, cm):
        cm.record_sensory("text", "will expire", ttl_seconds=0)
        cm.record_sensory("text", "will survive", ttl_seconds=600)
        time.sleep(0.1)
        pruned = cm.prune_expired_sensory()
        assert pruned == 1
        items = cm.get_recent_sensory()
        assert len(items) == 1
        assert items[0].raw_data == "will survive"

    def test_attend_to_sensory_promotes_to_episode(self, cm):
        sid = cm.record_sensory("error", "segfault detected", ttl_seconds=600)
        ep_id = cm.attend_to_sensory(sid, "critical error")
        assert ep_id is not None
        assert ep_id.startswith("epi_")

        episodes = cm.get_episodes()
        assert len(episodes) == 1
        assert "segfault detected" in episodes[0].content
        assert "critical error" in episodes[0].content

    def test_attend_to_missing_returns_none(self, cm):
        result = cm.attend_to_sensory("nonexistent", "reason")
        assert result is None


# ==================================================================
# WORKING MEMORY
# ==================================================================


class TestWorkingMemory:
    def test_push_and_get(self, cm):
        nid = cm.push_working("goal", "finish report", "task-1")
        assert nid.startswith("wrk_")

        slots = cm.get_working("task-1")
        assert len(slots) == 1
        assert slots[0].content == "finish report"
        assert slots[0].task_id == "task-1"

    def test_relevance_ordering(self, cm):
        cm.push_working("a", "low", "t1", relevance=0.1)
        cm.push_working("b", "high", "t1", relevance=0.9)
        cm.push_working("c", "mid", "t1", relevance=0.5)

        slots = cm.get_working("t1")
        assert slots[0].content == "high"
        assert slots[-1].content == "low"

    def test_capacity_eviction(self, cm):
        # Fill to capacity
        for i in range(CognitiveMemory.WORKING_MEMORY_CAPACITY):
            cm.push_working("item", f"slot-{i}", "t1", relevance=float(i))

        slots = cm.get_working("t1")
        assert len(slots) == CognitiveMemory.WORKING_MEMORY_CAPACITY

        # Push one more -- the slot with relevance 0.0 should be evicted
        cm.push_working("item", "new-slot", "t1", relevance=100.0)
        slots = cm.get_working("t1")
        assert len(slots) == CognitiveMemory.WORKING_MEMORY_CAPACITY
        contents = {s.content for s in slots}
        assert "new-slot" in contents
        assert "slot-0" not in contents  # lowest relevance evicted

    def test_clear_working(self, cm):
        cm.push_working("a", "x", "t1")
        cm.push_working("b", "y", "t1")
        cm.push_working("c", "z", "t2")

        cleared = cm.clear_working("t1")
        assert cleared == 2
        assert len(cm.get_working("t1")) == 0
        assert len(cm.get_working("t2")) == 1

    def test_task_isolation(self, cm):
        cm.push_working("a", "for t1", "t1")
        cm.push_working("b", "for t2", "t2")

        assert len(cm.get_working("t1")) == 1
        assert len(cm.get_working("t2")) == 1


# ==================================================================
# EPISODIC MEMORY
# ==================================================================


class TestEpisodicMemory:
    def test_store_and_retrieve(self, cm):
        eid = cm.store_episode("User asked about weather", "user-session")
        assert eid.startswith("epi_")

        episodes = cm.get_episodes()
        assert len(episodes) == 1
        assert episodes[0].content == "User asked about weather"
        assert episodes[0].source_label == "user-session"
        assert episodes[0].compressed is False

    def test_temporal_ordering(self, cm):
        cm.store_episode("first", "s")
        cm.store_episode("second", "s")
        cm.store_episode("third", "s")

        episodes = cm.get_episodes()
        assert episodes[0].content == "third"
        assert episodes[2].content == "first"

    def test_explicit_temporal_index(self, cm):
        cm.store_episode("a", "s", temporal_index=100)
        cm.store_episode("b", "s", temporal_index=50)

        episodes = cm.get_episodes()
        assert episodes[0].temporal_index == 100
        assert episodes[1].temporal_index == 50

    def test_metadata_roundtrip(self, cm):
        cm.store_episode("meta test", "s", metadata={"key": "value", "n": 42})
        ep = cm.get_episodes()[0]
        assert ep.metadata == {"key": "value", "n": 42}

    def test_consolidation(self, cm):
        # Store exactly batch_size episodes
        for i in range(5):
            cm.store_episode(f"episode-{i}", "batch-test")

        cons_id = cm.consolidate_episodes(batch_size=5)
        assert cons_id is not None
        assert cons_id.startswith("con_")

        # Compressed episodes excluded by default
        remaining = cm.get_episodes(include_compressed=False)
        assert len(remaining) == 0

        # Including compressed shows them
        all_eps = cm.get_episodes(include_compressed=True)
        assert len(all_eps) == 5
        assert all(e.compressed for e in all_eps)

    def test_consolidation_with_custom_summarizer(self, cm):
        for i in range(3):
            cm.store_episode(f"ep-{i}", "s")

        def my_summarizer(texts):
            return f"Summary of {len(texts)} items"

        cons_id = cm.consolidate_episodes(batch_size=3, summarizer=my_summarizer)
        assert cons_id is not None

    def test_consolidation_returns_none_when_insufficient(self, cm):
        cm.store_episode("only one", "s")
        assert cm.consolidate_episodes(batch_size=5) is None


# ==================================================================
# SEMANTIC MEMORY
# ==================================================================


class TestSemanticMemory:
    def test_store_and_retrieve(self, cm):
        fid = cm.store_fact("Python", "Python is a programming language", confidence=0.95)
        assert fid.startswith("sem_")

        facts = cm.get_all_facts()
        assert len(facts) == 1
        assert facts[0].concept == "Python"
        assert facts[0].confidence == 0.95

    def test_search_by_keyword(self, cm):
        cm.store_fact("Python", "Python is interpreted", confidence=0.9)
        cm.store_fact("Rust", "Rust is compiled", confidence=0.8)
        cm.store_fact("Go", "Go has goroutines", confidence=0.7)

        results = cm.search_facts("Python")
        assert len(results) >= 1
        assert results[0].concept == "Python"

    def test_search_min_confidence(self, cm):
        cm.store_fact("A", "low conf fact", confidence=0.2)
        cm.store_fact("B", "high conf fact", confidence=0.9)

        results = cm.search_facts("fact", min_confidence=0.5)
        assert len(results) == 1
        assert results[0].concept == "B"

    def test_tags_roundtrip(self, cm):
        cm.store_fact("X", "tagged fact", tags=["tag1", "tag2"])
        fact = cm.get_all_facts()[0]
        assert fact.tags == ["tag1", "tag2"]

    def test_metadata_roundtrip(self, cm):
        cm.store_fact("X", "meta fact", temporal_metadata={"year": 2025})
        fact = cm.get_all_facts()[0]
        assert fact.metadata == {"year": 2025}


# ==================================================================
# PROCEDURAL MEMORY
# ==================================================================


class TestProceduralMemory:
    def test_store_and_recall(self, cm):
        pid = cm.store_procedure(
            name="deploy to prod",
            steps=["run tests", "build image", "push to registry", "deploy"],
            prerequisites=["all tests pass"],
        )
        assert pid.startswith("proc_")

        procs = cm.recall_procedure("deploy")
        assert len(procs) == 1
        assert procs[0].name == "deploy to prod"
        assert len(procs[0].steps) == 4
        assert procs[0].prerequisites == ["all tests pass"]

    def test_usage_count_increments(self, cm):
        cm.store_procedure("test proc", ["step1"])

        # First recall
        procs = cm.recall_procedure("test proc")
        assert len(procs) == 1
        # After recall, usage_count in DB is incremented
        # The returned object still has the pre-increment value (0)

        # Second recall should show incremented count
        procs2 = cm.recall_procedure("test proc")
        assert procs2[0].usage_count == 1  # was 0, incremented by first recall

    def test_empty_query_returns_most_used(self, cm):
        cm.store_procedure("often used", ["a"])
        cm.store_procedure("rarely used", ["b"])

        # Recall "often used" to bump its count
        cm.recall_procedure("often used")

        procs = cm.recall_procedure("")
        assert procs[0].name == "often used"


# ==================================================================
# PROSPECTIVE MEMORY
# ==================================================================


class TestProspectiveMemory:
    def test_store_and_check_trigger(self, cm):
        nid = cm.store_prospective(
            description="Remind about meeting",
            trigger_condition="meeting calendar",
            action_on_trigger="Send reminder email",
            priority=5,
        )
        assert nid.startswith("pro_")

        triggered = cm.check_triggers("Check the calendar for today's meeting")
        assert len(triggered) == 1
        assert triggered[0].status == "triggered"
        assert triggered[0].description == "Remind about meeting"

    def test_no_trigger_on_unrelated_content(self, cm):
        cm.store_prospective(
            description="Deploy when ready",
            trigger_condition="deployment approval",
            action_on_trigger="Run deploy script",
        )

        triggered = cm.check_triggers("The weather is nice today")
        assert len(triggered) == 0

    def test_resolve_prospective(self, cm):
        nid = cm.store_prospective(
            description="Test resolve",
            trigger_condition="trigger word",
            action_on_trigger="do something",
        )

        # Trigger it
        cm.check_triggers("the trigger word appeared")

        # Resolve it
        cm.resolve_prospective(nid)

        # Should not trigger again
        triggered = cm.check_triggers("trigger word again")
        assert len(triggered) == 0

    def test_priority_ordering(self, cm):
        cm.store_prospective("low", "alpha", "act-low", priority=1)
        cm.store_prospective("high", "alpha", "act-high", priority=10)

        triggered = cm.check_triggers("alpha word")
        assert len(triggered) == 2
        assert triggered[0].priority > triggered[1].priority


# ==================================================================
# STATISTICS
# ==================================================================


class TestStatistics:
    def test_empty_stats(self, cm):
        stats = cm.get_statistics()
        assert stats["total"] == 0
        assert stats["sensory"] == 0
        assert stats["working"] == 0
        assert stats["episodic"] == 0
        assert stats["semantic"] == 0
        assert stats["procedural"] == 0
        assert stats["prospective"] == 0

    def test_counts_after_inserts(self, cm):
        cm.record_sensory("text", "hello")
        cm.push_working("goal", "x", "t1")
        cm.store_episode("episode", "s")
        cm.store_fact("concept", "content")
        cm.store_procedure("proc", ["step"])
        cm.store_prospective("desc", "trig", "act")

        stats = cm.get_statistics()
        assert stats["total"] == 6
        assert stats["sensory"] == 1
        assert stats["working"] == 1
        assert stats["episodic"] == 1
        assert stats["semantic"] == 1
        assert stats["procedural"] == 1
        assert stats["prospective"] == 1


# ==================================================================
# AGENT ISOLATION
# ==================================================================


class TestAgentIsolation:
    def test_different_agents_see_own_data(self, tmp_path):
        db = tmp_path / "shared_db"
        agent_a = CognitiveMemory(agent_name="agent-a", db_path=db)
        agent_b = CognitiveMemory(agent_name="agent-b", db_path=db)

        agent_a.store_fact("A-concept", "A-content")
        agent_b.store_fact("B-concept", "B-content")

        a_facts = agent_a.get_all_facts()
        b_facts = agent_b.get_all_facts()

        assert len(a_facts) == 1
        assert a_facts[0].concept == "A-concept"
        assert len(b_facts) == 1
        assert b_facts[0].concept == "B-concept"

        agent_a.close()
        agent_b.close()

    def test_sensory_isolation(self, tmp_path):
        db = tmp_path / "shared_db"
        a = CognitiveMemory(agent_name="a", db_path=db)
        b = CognitiveMemory(agent_name="b", db_path=db)

        a.record_sensory("text", "a-data")
        b.record_sensory("text", "b-data")

        assert len(a.get_recent_sensory()) == 1
        assert a.get_recent_sensory()[0].raw_data == "a-data"
        assert len(b.get_recent_sensory()) == 1
        assert b.get_recent_sensory()[0].raw_data == "b-data"

        a.close()
        b.close()

    def test_working_memory_isolation(self, tmp_path):
        db = tmp_path / "shared_db"
        a = CognitiveMemory(agent_name="a", db_path=db)
        b = CognitiveMemory(agent_name="b", db_path=db)

        a.push_working("goal", "a-goal", "t1")
        b.push_working("goal", "b-goal", "t1")

        assert len(a.get_working("t1")) == 1
        assert len(b.get_working("t1")) == 1

        a.close()
        b.close()

    def test_episode_isolation(self, tmp_path):
        db = tmp_path / "shared_db"
        a = CognitiveMemory(agent_name="a", db_path=db)
        b = CognitiveMemory(agent_name="b", db_path=db)

        a.store_episode("a-ep", "s")
        b.store_episode("b-ep", "s")

        assert len(a.get_episodes()) == 1
        assert len(b.get_episodes()) == 1

        a.close()
        b.close()

    def test_stats_isolation(self, tmp_path):
        db = tmp_path / "shared_db"
        a = CognitiveMemory(agent_name="a", db_path=db)
        b = CognitiveMemory(agent_name="b", db_path=db)

        a.store_fact("x", "y")
        a.store_fact("x2", "y2")
        b.store_fact("z", "w")

        assert a.get_statistics()["semantic"] == 2
        assert b.get_statistics()["semantic"] == 1

        a.close()
        b.close()


# ==================================================================
# ENTITY EXTRACTION AND RETRIEVAL
# ==================================================================


class TestEntityExtraction:
    """Tests for _extract_entity_name and entity-centric retrieval."""

    def test_extract_multi_word_name(self, cm):
        assert cm._extract_entity_name("", "Sarah Chen") == "sarah chen"

    def test_extract_apostrophe_name(self, cm):
        result = cm._extract_entity_name("Dr. O'Brien is a researcher", "")
        assert "o'brien" in result

    def test_extract_hyphenated_name(self, cm):
        result = cm._extract_entity_name("Al-Hassan won the award", "")
        assert "al-hassan" in result

    def test_extract_from_concept_first(self, cm):
        result = cm._extract_entity_name("some content", "Johannes Klaebo")
        assert result == "johannes klaebo"

    def test_extract_empty_returns_empty(self, cm):
        assert cm._extract_entity_name("", "") == ""

    def test_extract_no_proper_noun(self, cm):
        # No capitalized word after position 0
        assert cm._extract_entity_name("all lowercase text here", "") == ""

    def test_store_fact_sets_entity_name(self, cm):
        cm.store_fact("Sarah Chen", "Sarah Chen is a biologist", confidence=0.9)
        facts = cm.get_all_facts()
        assert len(facts) == 1
        assert facts[0].entity_name == "sarah chen"

    def test_retrieve_by_entity(self, cm):
        cm.store_fact("Sarah Chen", "Sarah Chen is a biologist", confidence=0.9)
        cm.store_fact("Sarah Chen hobbies", "Sarah Chen enjoys hiking", confidence=0.8)
        cm.store_fact("Bob Smith", "Bob Smith is an engineer", confidence=0.7)

        results = cm.retrieve_by_entity("Sarah Chen")
        assert len(results) == 2
        concepts = {r.concept for r in results}
        assert "Sarah Chen" in concepts or "Sarah Chen hobbies" in concepts

    def test_retrieve_by_entity_case_insensitive(self, cm):
        cm.store_fact("Sarah Chen", "Sarah Chen is a biologist", confidence=0.9)
        results = cm.retrieve_by_entity("sarah chen")
        assert len(results) == 1

    def test_retrieve_by_entity_fallback_to_content(self, cm):
        # Store a fact where the entity name extraction might not work
        # but the entity appears in content/concept
        cm.store_fact("biology", "study of living organisms", confidence=0.8)
        results = cm.retrieve_by_entity("biology")
        assert len(results) == 1

    def test_retrieve_by_entity_empty_returns_empty(self, cm):
        assert cm.retrieve_by_entity("") == []
        assert cm.retrieve_by_entity("   ") == []


# ==================================================================
# CONCEPT SEARCH
# ==================================================================


class TestConceptSearch:
    def test_search_by_concept_basic(self, cm):
        cm.store_fact("Python", "Python is interpreted", confidence=0.9)
        cm.store_fact("Rust", "Rust is compiled", confidence=0.8)
        cm.store_fact("Go", "Go has goroutines", confidence=0.7)

        results = cm.search_by_concept(["Python"])
        assert len(results) >= 1
        assert any("Python" in r.concept or "python" in r.content.lower() for r in results)

    def test_search_by_concept_multiple_keywords(self, cm):
        cm.store_fact("Python", "Python is interpreted", confidence=0.9)
        cm.store_fact("Rust", "Rust is compiled", confidence=0.8)

        results = cm.search_by_concept(["Python", "Rust"])
        assert len(results) == 2

    def test_search_by_concept_deduplicates(self, cm):
        cm.store_fact("Python language", "Python is popular", confidence=0.9)
        # "Python" and "language" should both match the same fact
        results = cm.search_by_concept(["Python", "language"])
        assert len(results) == 1

    def test_search_by_concept_skips_short_keywords(self, cm):
        cm.store_fact("Python", "Python is interpreted", confidence=0.9)
        results = cm.search_by_concept(["it", "is"])
        assert len(results) == 0  # "it" and "is" are <= 2 chars

    def test_search_by_concept_empty_returns_empty(self, cm):
        assert cm.search_by_concept([]) == []


# ==================================================================
# AGGREGATION QUERIES
# ==================================================================


class TestAggregation:
    def test_count_total(self, cm):
        cm.store_fact("A", "fact 1")
        cm.store_fact("B", "fact 2")
        cm.store_fact("C", "fact 3")

        result = cm.execute_aggregation("count_total")
        assert result["count"] == 3
        assert result["query_type"] == "count_total"

    def test_count_total_empty(self, cm):
        result = cm.execute_aggregation("count_total")
        assert result["count"] == 0

    def test_count_entities(self, cm):
        cm.store_fact("Sarah Chen", "Sarah Chen is a biologist", confidence=0.9)
        cm.store_fact("Bob Smith", "Bob Smith is an engineer", confidence=0.8)
        cm.store_fact("generic", "no entity here", confidence=0.7)

        result = cm.execute_aggregation("count_entities")
        assert result["count"] >= 2

    def test_list_entities(self, cm):
        cm.store_fact("Sarah Chen", "Sarah Chen is a biologist", confidence=0.9)
        cm.store_fact("Bob Smith", "Bob Smith is an engineer", confidence=0.8)

        result = cm.execute_aggregation("list_entities")
        assert result["count"] >= 2
        assert "items" in result

    def test_count_concepts(self, cm):
        cm.store_fact("Python", "Python is interpreted")
        cm.store_fact("Rust", "Rust is compiled")
        cm.store_fact("Go", "Go has goroutines")

        result = cm.execute_aggregation("count_concepts")
        assert result["count"] == 3

    def test_list_concepts(self, cm):
        cm.store_fact("Python", "Python is interpreted")
        cm.store_fact("Rust", "Rust is compiled")

        result = cm.execute_aggregation("list_concepts")
        assert result["count"] == 2
        assert "Python" in result["items"]
        assert "Rust" in result["items"]

    def test_list_concepts_with_filter(self, cm):
        cm.store_fact("Python", "Python is interpreted")
        cm.store_fact("Rust", "Rust is compiled")

        result = cm.execute_aggregation("list_concepts", entity_filter="python")
        assert result["count"] == 1
        assert "Python" in result["items"]

    def test_count_by_concept(self, cm):
        cm.store_fact("Python", "fact 1")
        cm.store_fact("Python", "fact 2")
        cm.store_fact("Rust", "fact 3")

        result = cm.execute_aggregation("count_by_concept")
        assert result["count"] == 2
        assert result["items"]["Python"] == 2
        assert result["items"]["Rust"] == 1
        assert result["total_facts"] == 3

    def test_unknown_query_type_returns_error(self, cm):
        result = cm.execute_aggregation("unknown_type")
        assert result["count"] == 0
        assert "error" in result


# ==================================================================
# SUPERSEDES DETECTION
# ==================================================================


class TestSupersedesDetection:
    def test_detect_contradiction(self, cm):
        result = cm._detect_contradiction(
            "Klaebo has 10 golds", "Klaebo has 9 golds",
            "Klaebo medals", "Klaebo medals",
        )
        assert result.get("contradiction") is True
        assert "conflicting_values" in result

    def test_no_contradiction_same_numbers(self, cm):
        result = cm._detect_contradiction(
            "Klaebo has 10 golds", "Klaebo has 10 golds",
            "Klaebo medals", "Klaebo medals",
        )
        assert result == {}

    def test_no_contradiction_different_concepts(self, cm):
        result = cm._detect_contradiction(
            "Klaebo has 10 golds", "Python version 3",
            "Klaebo", "Python",
        )
        assert result == {}

    def test_supersedes_edge_created(self, cm):
        # Store old fact with temporal_index=1
        cm.store_fact(
            "Klaebo medals", "Klaebo has 9 golds",
            temporal_metadata={"temporal_index": 1},
        )
        # Store new fact with temporal_index=2 (should supersede)
        cm.store_fact(
            "Klaebo medals", "Klaebo has 10 golds",
            temporal_metadata={"temporal_index": 2},
        )

        # Check that SUPERSEDES edge exists
        result = cm._conn.execute(
            """
            MATCH (a:SemanticMemory)-[r:SUPERSEDES]->(b:SemanticMemory)
            WHERE a.agent_id = $aid
            RETURN a.content, b.content, r.reason
            """,
            {"aid": cm.agent_name},
        )
        rows = []
        while result.has_next():
            rows.append(result.get_next())
        assert len(rows) >= 1
        # The newer fact should supersede the older one
        assert "10" in rows[0][0]  # new fact
        assert "9" in rows[0][1]   # old fact


# ==================================================================
# BACKWARD COMPATIBILITY
# ==================================================================


class TestBackwardCompatibility:
    """Verify that existing ExperienceStore and MemoryConnector still work."""

    def test_experience_store_still_works(self, tmp_path):
        from amplihack_memory import Experience, ExperienceStore, ExperienceType

        store = ExperienceStore(
            agent_name="compat-test",
            storage_path=tmp_path / "compat_store",
            backend="kuzu",
        )
        exp = Experience(
            experience_type=ExperienceType.SUCCESS,
            context="test context",
            outcome="test outcome",
            confidence=0.8,
        )
        exp_id = store.add(exp)
        assert exp_id is not None

    def test_memory_connector_still_works(self, tmp_path):
        from amplihack_memory import Experience, ExperienceType, MemoryConnector

        conn = MemoryConnector(
            agent_name="compat-test",
            storage_path=tmp_path / "compat_conn",
            backend="kuzu",
        )
        exp = Experience(
            experience_type=ExperienceType.INSIGHT,
            context="insight context",
            outcome="insight outcome",
            confidence=0.7,
        )
        exp_id = conn.store_experience(exp)
        assert exp_id is not None
        conn.close()


# ==================================================================
# EXPORT / IMPORT
# ==================================================================


class TestExportImport:
    """Tests for export_to_json() and import_from_json()."""

    def test_export_empty_memory(self, cm):
        data = cm.export_to_json()
        assert data["agent_name"] == "test-agent"
        assert data["format_version"] == "1.0"
        assert data["statistics"]["total_nodes"] == 0
        assert data["statistics"]["total_edges"] == 0
        for key in [
            "semantic_nodes", "episodic_nodes", "procedural_nodes",
            "prospective_nodes", "sensory_nodes", "working_nodes",
            "consolidated_nodes",
        ]:
            assert data[key] == []

    def test_export_semantic_nodes(self, cm):
        cm.store_fact("Biology", "Photosynthesis converts light", confidence=0.9)
        cm.store_fact("Physics", "E equals mc squared", confidence=0.95)

        data = cm.export_to_json()
        assert len(data["semantic_nodes"]) == 2
        assert data["statistics"]["semantic_count"] == 2
        # Verify fields are present
        node = data["semantic_nodes"][0]
        assert "node_id" in node
        assert "concept" in node
        assert "content" in node
        assert "confidence" in node
        assert "entity_name" in node

    def test_export_episodic_nodes(self, cm):
        cm.store_episode("User asked about photosynthesis", "session-1")
        cm.store_episode("System analyzed code", "session-2")

        data = cm.export_to_json()
        assert len(data["episodic_nodes"]) == 2
        assert data["statistics"]["episodic_count"] == 2

    def test_export_procedural_nodes(self, cm):
        cm.store_procedure("Deploy service", ["build", "test", "push", "deploy"])

        data = cm.export_to_json()
        assert len(data["procedural_nodes"]) == 1
        assert data["procedural_nodes"][0]["name"] == "Deploy service"
        assert len(data["procedural_nodes"][0]["steps"]) == 4

    def test_export_prospective_nodes(self, cm):
        cm.store_prospective(
            "Remind about tests",
            "code change detected",
            "run test suite",
            priority=2,
        )

        data = cm.export_to_json()
        assert len(data["prospective_nodes"]) == 1
        assert data["prospective_nodes"][0]["description"] == "Remind about tests"
        assert data["prospective_nodes"][0]["priority"] == 2

    def test_export_sensory_nodes(self, cm):
        cm.record_sensory("text", "Hello world", ttl_seconds=3600)

        data = cm.export_to_json()
        assert len(data["sensory_nodes"]) == 1
        assert data["sensory_nodes"][0]["modality"] == "text"

    def test_export_working_nodes(self, cm):
        cm.push_working("goal", "Implement auth", task_id="task-1")

        data = cm.export_to_json()
        assert len(data["working_nodes"]) == 1
        assert data["working_nodes"][0]["task_id"] == "task-1"

    def test_export_supersedes_edges(self, cm):
        cm.store_fact(
            "Klaebo medals",
            "Klaebo has 9 gold medals",
            temporal_metadata={"temporal_index": 1},
        )
        cm.store_fact(
            "Klaebo medals",
            "Klaebo has 10 gold medals",
            temporal_metadata={"temporal_index": 2},
        )

        data = cm.export_to_json()
        assert len(data["supersedes_edges"]) == 1

    def test_roundtrip_semantic(self, cm, tmp_path):
        """Export from one instance, import into another, verify data."""
        # Populate source
        cm.store_fact("Math", "Pi is approximately 3.14159", confidence=0.99)
        cm.store_fact("History", "Rome was founded in 753 BC", confidence=0.85)
        cm.store_episode("Session started", "test-session")

        # Export
        data = cm.export_to_json()

        # Create a new instance and import
        target_db = tmp_path / "import_target"
        target = CognitiveMemory(agent_name="imported-agent", db_path=target_db)
        try:
            stats = target.import_from_json(data)
            assert stats["nodes_imported"] == 3  # 2 semantic + 1 episodic
            assert stats["errors"] == 0
            assert stats["skipped"] == 0

            # Verify data exists in target
            facts = target.get_all_facts(limit=10)
            assert len(facts) == 2
            contents = {f.content for f in facts}
            assert "Pi is approximately 3.14159" in contents
            assert "Rome was founded in 753 BC" in contents
        finally:
            target.close()

    def test_import_merge_mode(self, cm, tmp_path):
        """Merge import should skip existing nodes and add new ones."""
        # Store initial fact
        cm.store_fact("Science", "Water boils at 100C")

        # Create export data with different facts
        export_data = {
            "format_version": "1.0",
            "semantic_nodes": [
                {
                    "node_id": "new-fact-1",
                    "concept": "Chemistry",
                    "content": "H2O is water",
                    "confidence": 0.9,
                    "source_id": "",
                    "tags": [],
                    "metadata": {},
                    "created_at": 1000,
                    "entity_name": "",
                },
            ],
            "episodic_nodes": [],
        }

        stats = cm.import_from_json(export_data, merge=True)
        assert stats["nodes_imported"] == 1
        assert stats["skipped"] == 0

        # Should have 2 facts now
        facts = cm.get_all_facts(limit=10)
        assert len(facts) == 2

    def test_import_replace_mode(self, cm):
        """Non-merge import should clear existing data first."""
        # Store initial facts
        cm.store_fact("Science", "Water boils at 100C")
        cm.store_fact("Science", "Ice melts at 0C")

        # Import with merge=False (replace)
        export_data = {
            "format_version": "1.0",
            "semantic_nodes": [
                {
                    "node_id": "replacement-1",
                    "concept": "Music",
                    "content": "Bach was a composer",
                    "confidence": 0.95,
                    "source_id": "",
                    "tags": [],
                    "metadata": {},
                    "created_at": 2000,
                    "entity_name": "bach",
                },
            ],
            "episodic_nodes": [],
        }

        stats = cm.import_from_json(export_data, merge=False)
        assert stats["nodes_imported"] == 1

        # Should have only 1 fact (the old ones were cleared)
        facts = cm.get_all_facts(limit=10)
        assert len(facts) == 1
        assert facts[0].content == "Bach was a composer"

    def test_import_handles_missing_keys(self, cm):
        """Import should handle partial data gracefully."""
        export_data = {
            "format_version": "1.0",
            "semantic_nodes": [
                {
                    "node_id": "sparse-1",
                    "concept": "Test",
                    "content": "Sparse data",
                    "confidence": 0.5,
                    "created_at": 3000,
                },
            ],
        }

        stats = cm.import_from_json(export_data, merge=False)
        assert stats["nodes_imported"] == 1
        assert stats["errors"] == 0

    def test_import_skips_empty_node_ids(self, cm):
        """Nodes without node_id should be counted as errors."""
        export_data = {
            "format_version": "1.0",
            "semantic_nodes": [
                {"concept": "Bad", "content": "No ID"},  # missing node_id
            ],
            "episodic_nodes": [],
        }

        stats = cm.import_from_json(export_data, merge=False)
        assert stats["errors"] == 1
        assert stats["nodes_imported"] == 0

    def test_export_statistics_correct(self, cm):
        """Verify statistics block counts match actual data."""
        cm.store_fact("A", "fact 1")
        cm.store_fact("B", "fact 2")
        cm.store_episode("ep 1", "src-1")
        cm.store_procedure("proc 1", ["step1"])
        cm.store_prospective("remind", "trigger", "action")
        cm.record_sensory("text", "observation")
        cm.push_working("goal", "do thing", task_id="t1")

        data = cm.export_to_json()
        s = data["statistics"]
        assert s["semantic_count"] == 2
        assert s["episodic_count"] == 1
        assert s["procedural_count"] == 1
        assert s["prospective_count"] == 1
        assert s["sensory_count"] == 1
        assert s["working_count"] == 1
        assert s["total_nodes"] == 7

    def test_roundtrip_all_node_types(self, cm, tmp_path):
        """Full roundtrip of all 6 memory types + consolidated episodes."""
        # Store one of each type
        cm.store_fact("Biology", "Cells divide")
        cm.store_episode("Session happened", "src")
        cm.store_procedure("Deploy", ["build", "test"])
        cm.store_prospective("reminder", "trigger", "action")
        cm.record_sensory("text", "raw data", ttl_seconds=9999)
        cm.push_working("goal", "task content", task_id="t1")

        # Export
        data = cm.export_to_json()
        assert data["statistics"]["total_nodes"] == 6

        # Import into new instance
        target_db = tmp_path / "full_roundtrip"
        target = CognitiveMemory(agent_name="roundtrip-agent", db_path=target_db)
        try:
            stats = target.import_from_json(data)
            assert stats["nodes_imported"] == 6
            assert stats["errors"] == 0

            # Verify each type
            target_stats = target.get_statistics()
            assert target_stats["semantic"] == 1
            assert target_stats["episodic"] == 1
            assert target_stats["procedural"] == 1
            assert target_stats["prospective"] == 1
            assert target_stats["sensory"] == 1
            assert target_stats["working"] == 1
        finally:
            target.close()
