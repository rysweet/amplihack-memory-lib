"""Tests for HierarchicalMemory in amplihack-memory-lib.

Tests the full HierarchicalMemory class including store, retrieve, subgraph,
entity search, export/import, and protocol-compatible aliases.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from amplihack_memory.hierarchical_memory import (
    HierarchicalMemory,
    KnowledgeEdge,
    KnowledgeNode,
    KnowledgeSubgraph,
    MemoryCategory,
    MemoryClassifier,
)


@pytest.fixture()
def mem(tmp_path: Path) -> HierarchicalMemory:
    """Create a fresh HierarchicalMemory instance for testing."""
    m = HierarchicalMemory("test_agent", db_path=tmp_path / "test_db")
    yield m
    m.close()


class TestMemoryCategory:
    def test_categories_exist(self):
        assert MemoryCategory.EPISODIC.value == "episodic"
        assert MemoryCategory.SEMANTIC.value == "semantic"
        assert MemoryCategory.PROCEDURAL.value == "procedural"
        assert MemoryCategory.PROSPECTIVE.value == "prospective"
        assert MemoryCategory.WORKING.value == "working"


class TestMemoryClassifier:
    def test_procedural_classification(self):
        c = MemoryClassifier()
        assert c.classify("Follow these steps to bake a cake") == MemoryCategory.PROCEDURAL

    def test_prospective_classification(self):
        c = MemoryClassifier()
        assert c.classify("I plan to visit Paris next year") == MemoryCategory.PROSPECTIVE

    def test_episodic_classification(self):
        c = MemoryClassifier()
        assert c.classify("I saw a strange event yesterday") == MemoryCategory.EPISODIC

    def test_semantic_default(self):
        c = MemoryClassifier()
        # Note: "plants" contains "plan" as substring, triggering PROSPECTIVE.
        # Use text without substring matches to verify SEMANTIC default.
        assert c.classify("Water freezes at zero degrees celsius") == MemoryCategory.SEMANTIC


class TestKnowledgeSubgraph:
    def test_empty_subgraph_context(self):
        sg = KnowledgeSubgraph(query="test")
        assert sg.to_llm_context() == "No relevant knowledge found."

    def test_context_with_nodes(self):
        node = KnowledgeNode(
            node_id="abc",
            category=MemoryCategory.SEMANTIC,
            content="Plants use photosynthesis",
            concept="biology",
            confidence=0.9,
        )
        sg = KnowledgeSubgraph(nodes=[node], query="photosynthesis")
        ctx = sg.to_llm_context()
        assert "photosynthesis" in ctx
        assert "biology" in ctx

    def test_chronological_context(self):
        nodes = [
            KnowledgeNode(
                node_id="a",
                category=MemoryCategory.SEMANTIC,
                content="First fact",
                concept="test",
                confidence=0.5,
                metadata={"temporal_index": 1},
            ),
            KnowledgeNode(
                node_id="b",
                category=MemoryCategory.SEMANTIC,
                content="Second fact",
                concept="test",
                confidence=0.9,
                metadata={"temporal_index": 2},
            ),
        ]
        sg = KnowledgeSubgraph(nodes=nodes, query="test")
        ctx = sg.to_llm_context(chronological=True)
        first_idx = ctx.index("First fact")
        second_idx = ctx.index("Second fact")
        assert first_idx < second_idx

    def test_contradiction_warnings(self):
        nodes = [
            KnowledgeNode(
                node_id="a",
                category=MemoryCategory.SEMANTIC,
                content="Fact A",
                concept="test",
            ),
        ]
        edges = [
            KnowledgeEdge(
                source_id="a",
                target_id="b",
                relationship="SIMILAR_TO",
                metadata={"contradiction": True, "conflicting_values": "3 vs 5"},
            ),
        ]
        sg = KnowledgeSubgraph(nodes=nodes, edges=edges, query="test")
        ctx = sg.to_llm_context()
        assert "WARNING" in ctx
        assert "3 vs 5" in ctx


class TestHierarchicalMemoryInit:
    def test_empty_agent_name_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="cannot be empty"):
            HierarchicalMemory("", db_path=tmp_path / "db")

    def test_invalid_agent_name_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="alphanumeric"):
            HierarchicalMemory("../bad/path", db_path=tmp_path / "db")


class TestStoreAndRetrieve:
    def test_store_knowledge_returns_id(self, mem: HierarchicalMemory):
        nid = mem.store_knowledge("Plants use photosynthesis", "biology")
        assert nid is not None
        assert len(nid) > 0

    def test_store_empty_content_raises(self, mem: HierarchicalMemory):
        with pytest.raises(ValueError, match="cannot be empty"):
            mem.store_knowledge("")

    def test_store_and_retrieve_subgraph(self, mem: HierarchicalMemory):
        mem.store_knowledge("Photosynthesis converts light to energy", "biology")
        sg = mem.retrieve_subgraph("photosynthesis")
        assert len(sg.nodes) >= 1
        assert any("photosynthesis" in n.content.lower() for n in sg.nodes)

    def test_retrieve_empty_query(self, mem: HierarchicalMemory):
        sg = mem.retrieve_subgraph("")
        assert len(sg.nodes) == 0

    def test_store_episode_and_derives_from(self, mem: HierarchicalMemory):
        eid = mem.store_episode("Raw article about biology", "Wikipedia: Biology")
        nid = mem.store_knowledge("Cells are the basic unit of life", "biology", source_id=eid)
        assert nid is not None

    def test_store_with_temporal_metadata(self, mem: HierarchicalMemory):
        mem.store_knowledge(
            "Klaebo has 9 gold medals",
            "Klaebo medals",
            temporal_metadata={"temporal_index": 1, "source_date": "2024-01-01"},
        )
        mem.store_knowledge(
            "Klaebo has 10 gold medals",
            "Klaebo medals",
            temporal_metadata={"temporal_index": 2, "source_date": "2024-02-01"},
        )
        sg = mem.retrieve_subgraph("Klaebo medals")
        assert len(sg.nodes) >= 1


class TestProtocolAliases:
    """Test protocol-compatible method aliases."""

    def test_store_fact_delegates(self, mem: HierarchicalMemory):
        nid = mem.store_fact("Test fact content", "test concept")
        assert nid is not None
        assert len(nid) > 0

    def test_search_facts_returns_dicts(self, mem: HierarchicalMemory):
        mem.store_knowledge("Quantum physics is fascinating", "physics")
        results = mem.search_facts("quantum physics")
        assert isinstance(results, list)
        if results:
            r = results[0]
            assert "content" in r
            assert "concept" in r
            assert "confidence" in r

    def test_get_all_facts_returns_dicts(self, mem: HierarchicalMemory):
        mem.store_knowledge("Fact one", "concept one")
        mem.store_knowledge("Fact two", "concept two")
        results = mem.get_all_facts()
        assert isinstance(results, list)
        assert len(results) >= 2
        for r in results:
            assert "content" in r
            assert "concept" in r


class TestEntityRetrieval:
    def test_retrieve_by_entity(self, mem: HierarchicalMemory):
        mem.store_knowledge("Sarah Chen loves painting", "Sarah Chen hobbies")
        nodes = mem.retrieve_by_entity("Sarah Chen")
        assert len(nodes) >= 1
        assert any("sarah chen" in n.content.lower() for n in nodes)

    def test_retrieve_nonexistent_entity(self, mem: HierarchicalMemory):
        nodes = mem.retrieve_by_entity("NonexistentPerson")
        assert len(nodes) == 0


class TestSearchByConcept:
    def test_search_by_concept(self, mem: HierarchicalMemory):
        mem.store_knowledge("Water freezes at 0 degrees", "water properties")
        nodes = mem.search_by_concept(["water"])
        assert len(nodes) >= 1

    def test_search_empty_keywords(self, mem: HierarchicalMemory):
        assert mem.search_by_concept([]) == []


class TestAggregation:
    def test_count_total(self, mem: HierarchicalMemory):
        mem.store_knowledge("Fact 1", "topic")
        mem.store_knowledge("Fact 2", "topic")
        result = mem.execute_aggregation("count_total")
        assert result["count"] >= 2

    def test_list_entities(self, mem: HierarchicalMemory):
        mem.store_knowledge("Sarah Chen plays piano", "Sarah Chen music")
        result = mem.execute_aggregation("list_entities")
        assert "items" in result


class TestGetAllKnowledge:
    def test_returns_all(self, mem: HierarchicalMemory):
        mem.store_knowledge("Fact A", "topic_a")
        mem.store_knowledge("Fact B", "topic_b")
        nodes = mem.get_all_knowledge()
        assert len(nodes) >= 2

    def test_limit(self, mem: HierarchicalMemory):
        for i in range(5):
            mem.store_knowledge(f"Fact {i}", f"topic_{i}")
        nodes = mem.get_all_knowledge(limit=2)
        assert len(nodes) == 2


class TestStatistics:
    def test_get_statistics(self, mem: HierarchicalMemory):
        mem.store_knowledge("Test fact", "test")
        stats = mem.get_statistics()
        assert stats["agent_name"] == "test_agent"
        assert stats["semantic_nodes"] >= 1


class TestExportImport:
    def test_export_and_import(self, tmp_path: Path):
        # Create and populate source memory
        src = HierarchicalMemory("export_test", db_path=tmp_path / "src_db")
        src.store_knowledge("Fact alpha", "alpha")
        src.store_knowledge("Fact beta", "beta")
        exported = src.export_to_json()
        src.close()

        assert exported["statistics"]["semantic_node_count"] >= 2

        # Import into a new memory
        dst = HierarchicalMemory("import_test", db_path=tmp_path / "dst_db")
        stats = dst.import_from_json(exported)
        assert stats["semantic_nodes_imported"] >= 2
        assert stats["errors"] == 0

        # Verify data is there
        nodes = dst.get_all_knowledge()
        assert len(nodes) >= 2
        dst.close()

    def test_import_merge_dedup(self, tmp_path: Path):
        mem = HierarchicalMemory("merge_test", db_path=tmp_path / "merge_db")
        mem.store_knowledge("Original fact", "original")
        exported = mem.export_to_json()

        # Import same data again with merge=True
        stats = mem.import_from_json(exported, merge=True)
        assert stats["skipped"] >= 1
        mem.close()
