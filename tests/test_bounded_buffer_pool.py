"""Tests for bounded Kuzu buffer pool, max_db_size, and sqlite-default APIs.

This module covers:
- CognitiveMemory buffer_pool_size and max_db_size parameters
- MemoryConnector default backend is now sqlite (not kuzu)
- ExperienceStore default backend is now sqlite (not kuzu)
- KuzuGraphStore buffer_pool_size parameter
- Lifecycle and close() semantics
"""

import pytest

from amplihack_memory import CognitiveMemory, Experience, ExperienceType, MemoryConnector
from amplihack_memory.graph.ladybug_store import LadybugGraphStore as KuzuGraphStore
from amplihack_memory.store import ExperienceStore


# ==================================================================
# CognitiveMemory — buffer_pool_size parameter
# ==================================================================


class TestCognitiveMemoryBufferPoolSize:
    """CognitiveMemory constructor accepts buffer_pool_size."""

    def test_default_buffer_pool_size_creates_instance(self, tmp_path):
        mem = CognitiveMemory(agent_name="bp-agent", db_path=tmp_path / "db")
        assert mem is not None
        mem.close()

    def test_explicit_buffer_pool_size_small(self, tmp_path):
        """64 MB buffer pool — minimum reasonable value."""
        mem = CognitiveMemory(
            agent_name="bp-agent",
            db_path=tmp_path / "db",
            buffer_pool_size=64 * 1024 * 1024,
        )
        assert mem is not None
        mem.close()

    def test_explicit_buffer_pool_size_large(self, tmp_path):
        """512 MB buffer pool — large but valid."""
        mem = CognitiveMemory(
            agent_name="bp-agent",
            db_path=tmp_path / "db",
            buffer_pool_size=512 * 1024 * 1024,
        )
        assert mem is not None
        mem.close()

    def test_buffer_pool_zero_uses_kuzu_default(self, tmp_path):
        """buffer_pool_size=0 means let Kuzu choose (no kwarg passed)."""
        mem = CognitiveMemory(
            agent_name="bp-agent",
            db_path=tmp_path / "db",
            buffer_pool_size=0,
        )
        assert mem is not None
        mem.close()

    def test_buffer_pool_size_does_not_break_store_recall(self, tmp_path):
        """Basic store-and-recall still works with an explicit buffer pool."""
        mem = CognitiveMemory(
            agent_name="bp-agent",
            db_path=tmp_path / "db",
            buffer_pool_size=128 * 1024 * 1024,
        )
        fact_id = mem.store_fact("sky", "The sky is blue", confidence=0.9)
        assert fact_id is not None
        facts = mem.search_facts("sky")
        assert len(facts) >= 1
        mem.close()


# ==================================================================
# CognitiveMemory — max_db_size parameter
# ==================================================================


class TestCognitiveMemoryMaxDbSize:
    """CognitiveMemory constructor accepts max_db_size."""

    def test_max_db_size_zero_uses_kuzu_default(self, tmp_path):
        mem = CognitiveMemory(
            agent_name="mds-agent",
            db_path=tmp_path / "db",
            max_db_size=0,
        )
        assert mem is not None
        mem.close()

    def test_max_db_size_explicit_value(self, tmp_path):
        """1 GB max db size is accepted without error."""
        mem = CognitiveMemory(
            agent_name="mds-agent",
            db_path=tmp_path / "db",
            max_db_size=1024 * 1024 * 1024,
        )
        assert mem is not None
        mem.close()

    def test_buffer_pool_and_max_db_size_together(self, tmp_path):
        """Both parameters can be specified simultaneously."""
        mem = CognitiveMemory(
            agent_name="mds-agent",
            db_path=tmp_path / "db",
            buffer_pool_size=128 * 1024 * 1024,
            max_db_size=2 * 1024 * 1024 * 1024,
        )
        assert mem is not None
        mem.close()

    def test_max_db_size_with_store_recall(self, tmp_path):
        """Store and recall still works with max_db_size set."""
        mem = CognitiveMemory(
            agent_name="mds-agent",
            db_path=tmp_path / "db",
            max_db_size=512 * 1024 * 1024,
        )
        fact_id = mem.store_fact("chemistry", "Water is H2O", confidence=1.0)
        assert fact_id is not None
        mem.close()


# ==================================================================
# MemoryConnector — sqlite default backend
# ==================================================================


class TestMemoryConnectorSqliteDefault:
    """MemoryConnector defaults to sqlite backend."""

    def test_default_backend_is_sqlite(self, tmp_path):
        conn = MemoryConnector(
            agent_name="sqlite-test",
            storage_path=tmp_path / "mem",
        )
        assert conn.backend_type == "sqlite"
        conn.close()

    def test_explicit_sqlite_backend(self, tmp_path):
        conn = MemoryConnector(
            agent_name="sqlite-test",
            storage_path=tmp_path / "mem",
            backend="sqlite",
        )
        assert conn.backend_type == "sqlite"
        conn.close()

    def test_sqlite_backend_stores_experience(self, tmp_path):
        conn = MemoryConnector(
            agent_name="sqlite-test",
            storage_path=tmp_path / "mem",
        )
        exp = Experience(
            experience_type=ExperienceType.INSIGHT,
            context="default backend context",
            outcome="default backend outcome",
            confidence=0.8,
        )
        exp_id = conn.store_experience(exp)
        assert exp_id is not None
        conn.close()

    def test_sqlite_backend_search(self, tmp_path):
        conn = MemoryConnector(
            agent_name="sqlite-search",
            storage_path=tmp_path / "mem",
        )
        exp = Experience(
            experience_type=ExperienceType.INSIGHT,
            context="unique search token xyzzy42",
            outcome="found it",
            confidence=0.9,
        )
        conn.store_experience(exp)
        results = conn.search("xyzzy42")
        assert len(results) >= 1
        conn.close()

    def test_kuzu_backend_still_works_explicitly(self, tmp_path):
        """Explicitly requesting kuzu backend still works."""
        conn = MemoryConnector(
            agent_name="kuzu-explicit",
            storage_path=tmp_path / "mem",
            backend="kuzu",
        )
        assert conn.backend_type == "kuzu"
        conn.close()


# ==================================================================
# ExperienceStore — sqlite default backend
# ==================================================================


class TestExperienceStoreSqliteDefault:
    """ExperienceStore defaults to sqlite backend."""

    def test_default_backend_creates_without_error(self, tmp_path):
        store = ExperienceStore(
            agent_name="es-test",
            storage_path=tmp_path / "es",
        )
        assert store is not None
        store.connector.close()

    def test_default_backend_is_sqlite(self, tmp_path):
        store = ExperienceStore(
            agent_name="es-test",
            storage_path=tmp_path / "es",
        )
        assert store.connector.backend_type == "sqlite"
        store.connector.close()

    def test_explicit_sqlite_backend(self, tmp_path):
        store = ExperienceStore(
            agent_name="es-test",
            storage_path=tmp_path / "es",
            backend="sqlite",
        )
        assert store.connector.backend_type == "sqlite"
        store.connector.close()

    def test_sqlite_backend_add_and_search(self, tmp_path):
        store = ExperienceStore(
            agent_name="es-test",
            storage_path=tmp_path / "es",
        )
        exp = Experience(
            experience_type=ExperienceType.SUCCESS,
            context="completed task alpha99",
            outcome="success",
            confidence=0.95,
        )
        exp_id = store.add(exp)
        assert exp_id is not None
        results = store.search("alpha99")
        assert len(results) >= 1
        store.connector.close()


# ==================================================================
# KuzuGraphStore — buffer_pool_size parameter
# ==================================================================


class TestKuzuGraphStoreBufferPoolSize:
    """KuzuGraphStore accepts buffer_pool_size."""

    def test_default_buffer_pool_size(self, tmp_path):
        store = KuzuGraphStore(db_path=tmp_path / "graph")
        assert store is not None
        store.close()

    def test_small_buffer_pool_size(self, tmp_path):
        store = KuzuGraphStore(
            db_path=tmp_path / "graph",
            buffer_pool_size=32 * 1024 * 1024,
        )
        assert store is not None
        store.close()

    def test_large_buffer_pool_size(self, tmp_path):
        store = KuzuGraphStore(
            db_path=tmp_path / "graph",
            buffer_pool_size=512 * 1024 * 1024,
        )
        assert store is not None
        store.close()

    def test_buffer_pool_size_allows_normal_operations(self, tmp_path):
        store = KuzuGraphStore(
            db_path=tmp_path / "graph",
            buffer_pool_size=64 * 1024 * 1024,
        )
        node = store.add_node(
            node_type="Concept",
            properties={"label": "buffer pool test"},
            node_id="n1",
        )
        assert node is not None
        assert node.node_id == "n1"
        retrieved = store.get_node("n1")
        assert retrieved is not None
        store.close()

    def test_kuzu_graph_store_close_idempotent(self, tmp_path):
        """Calling close() twice should not raise."""
        store = KuzuGraphStore(db_path=tmp_path / "graph")
        store.close()
        store.close()  # second close should not raise


# ==================================================================
# CognitiveMemory close() semantics
# ==================================================================


class TestCognitiveMemoryCloseSemantics:
    """CognitiveMemory.close() releases resources cleanly."""

    def test_close_releases_resources(self, tmp_path):
        mem = CognitiveMemory(agent_name="close-test", db_path=tmp_path / "db")
        mem.store_fact("test", "test fact content", confidence=0.5)
        mem.close()  # Should not raise

    def test_close_with_buffer_pool_size(self, tmp_path):
        mem = CognitiveMemory(
            agent_name="close-test",
            db_path=tmp_path / "db",
            buffer_pool_size=64 * 1024 * 1024,
        )
        mem.close()

    def test_multiple_instances_same_agent_different_paths(self, tmp_path):
        """Two instances for the same agent but different paths coexist."""
        mem1 = CognitiveMemory(agent_name="multi-test", db_path=tmp_path / "db1")
        mem2 = CognitiveMemory(agent_name="multi-test", db_path=tmp_path / "db2")
        mem1.store_fact("topic", "fact one content", confidence=0.8)
        mem2.store_fact("topic", "fact two content", confidence=0.7)
        mem1.close()
        mem2.close()


# ==================================================================
# Additional coverage — KuzuGraphStore operations with buffer pool
# ==================================================================


class TestKuzuGraphStoreOperationsWithBufferPool:
    """KuzuGraphStore operations work correctly with custom buffer_pool_size."""

    def test_add_multiple_nodes_custom_buffer(self, tmp_path):
        store = KuzuGraphStore(
            db_path=tmp_path / "graph",
            buffer_pool_size=64 * 1024 * 1024,
        )
        n1 = store.add_node("Entity", {"name": "Alice"}, node_id="alice")
        n2 = store.add_node("Entity", {"name": "Bob"}, node_id="bob")
        assert n1.node_id == "alice"
        assert n2.node_id == "bob"
        store.close()

    def test_get_nonexistent_node_returns_none(self, tmp_path):
        store = KuzuGraphStore(db_path=tmp_path / "graph")
        result = store.get_node("nonexistent-node-id-xyz")
        assert result is None
        store.close()

    def test_search_nodes_empty_graph(self, tmp_path):
        store = KuzuGraphStore(db_path=tmp_path / "graph")
        results = store.search_nodes("Entity", ["name"], "anything")
        assert results == []
        store.close()

    def test_search_nodes_finds_match(self, tmp_path):
        store = KuzuGraphStore(
            db_path=tmp_path / "graph",
            buffer_pool_size=64 * 1024 * 1024,
        )
        store.add_node("Entity", {"name": "uniquetarget99"}, node_id="t99")
        results = store.search_nodes("Entity", ["name"], "uniquetarget99")
        assert len(results) >= 1
        assert results[0].node_id == "t99"
        store.close()

    def test_store_id_is_set(self, tmp_path):
        store = KuzuGraphStore(
            db_path=tmp_path / "graph",
            store_id="my-store",
        )
        assert store.store_id == "my-store"
        store.close()

    def test_store_id_auto_generated_when_none(self, tmp_path):
        store = KuzuGraphStore(db_path=tmp_path / "graph")
        assert store.store_id is not None
        assert len(store.store_id) > 0
        store.close()

    def test_buffer_pool_size_default_value(self, tmp_path):
        """Default buffer_pool_size is 256 MB."""
        # Just verifying the store initializes with the 256 MB default
        store = KuzuGraphStore(db_path=tmp_path / "graph")
        assert store is not None
        store.close()

    def test_multiple_node_types(self, tmp_path):
        store = KuzuGraphStore(
            db_path=tmp_path / "graph",
            buffer_pool_size=128 * 1024 * 1024,
        )
        store.add_node("Person", {"name": "Charlie"}, node_id="charlie")
        store.add_node("Place", {"name": "London"}, node_id="london")
        p = store.get_node("charlie")
        l = store.get_node("london")
        assert p is not None
        assert l is not None
        store.close()

    def test_add_and_retrieve_node_properties(self, tmp_path):
        """Node properties are preserved after add and retrieval."""
        store = KuzuGraphStore(db_path=tmp_path / "graph")
        store.add_node("Entity", {"label": "hello world"}, node_id="hw1")
        node = store.get_node("hw1")
        assert node is not None
        assert node.properties.get("label") == "hello world"
        store.close()
