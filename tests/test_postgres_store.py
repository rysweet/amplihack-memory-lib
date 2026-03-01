"""Tests for PostgresGraphStore and InMemoryGraphStore.

Tests use InMemoryGraphStore exclusively since PostgreSQL + AGE is not
guaranteed to be available.  The InMemoryGraphStore implements the same
interface, so protocol compliance tested here applies to both.

Test coverage:
    - Protocol compliance (isinstance check)
    - Node CRUD (add, get, query, search, update, delete)
    - Edge operations (add, query_neighbors, delete)
    - Traversal (1-hop, 2-hop, cross-boundary detection)
    - Deduplication in search results
    - Thread safety (concurrent operations)
    - Factory function (create_hive_store)
    - Edge cases (missing nodes, empty graphs, duplicate IDs)
"""

from __future__ import annotations

import threading

import pytest

from amplihack_memory.graph.postgres_store import (
    InMemoryGraphStore,
    PostgresGraphStore,
    create_hive_store,
)
from amplihack_memory.graph.protocol import GraphStore
from amplihack_memory.graph.types import (
    Direction,
    GraphEdge,
    GraphNode,
    TraversalResult,
)


@pytest.fixture
def store() -> InMemoryGraphStore:
    """Fresh InMemoryGraphStore for each test."""
    return InMemoryGraphStore(store_id="test-store")


# ── Protocol compliance ─────────────────────────────────────────────


class TestProtocolCompliance:
    """Verify InMemoryGraphStore satisfies the GraphStore protocol."""

    def test_isinstance_check(self, store: InMemoryGraphStore) -> None:
        assert isinstance(store, GraphStore)

    def test_store_id_property(self, store: InMemoryGraphStore) -> None:
        assert store.store_id == "test-store"

    def test_all_protocol_methods_exist(self, store: InMemoryGraphStore) -> None:
        required = [
            "add_node", "get_node", "query_nodes", "search_nodes",
            "update_node", "delete_node", "add_edge", "query_neighbors",
            "delete_edge", "traverse", "close",
        ]
        for method in required:
            assert hasattr(store, method), f"Missing method: {method}"
            assert callable(getattr(store, method)), f"Not callable: {method}"


# ── Node CRUD ───────────────────────────────────────────────────────


class TestNodeCRUD:
    """Node create, read, update, delete operations."""

    def test_add_node_returns_graph_node(self, store: InMemoryGraphStore) -> None:
        node = store.add_node("Agent", {"name": "alice"})
        assert isinstance(node, GraphNode)
        assert node.node_type == "Agent"
        assert node.properties["name"] == "alice"
        assert node.graph_origin == "test-store"

    def test_add_node_with_explicit_id(self, store: InMemoryGraphStore) -> None:
        node = store.add_node("Agent", {"name": "bob"}, node_id="bob-001")
        assert node.node_id == "bob-001"

    def test_add_node_generates_id_when_none(self, store: InMemoryGraphStore) -> None:
        node = store.add_node("Agent", {"name": "carol"})
        assert len(node.node_id) > 0

    def test_get_node_returns_node(self, store: InMemoryGraphStore) -> None:
        created = store.add_node("Agent", {"name": "dave"}, node_id="d1")
        fetched = store.get_node("d1")
        assert fetched is not None
        assert fetched.node_id == "d1"
        assert fetched.properties["name"] == "dave"

    def test_get_node_returns_none_for_missing(self, store: InMemoryGraphStore) -> None:
        assert store.get_node("nonexistent") is None

    def test_query_nodes_by_type(self, store: InMemoryGraphStore) -> None:
        store.add_node("Agent", {"name": "a1"})
        store.add_node("Agent", {"name": "a2"})
        store.add_node("Fact", {"content": "some fact"})
        agents = store.query_nodes("Agent")
        assert len(agents) == 2
        assert all(n.node_type == "Agent" for n in agents)

    def test_query_nodes_with_filters(self, store: InMemoryGraphStore) -> None:
        store.add_node("Agent", {"name": "alice", "role": "admin"})
        store.add_node("Agent", {"name": "bob", "role": "user"})
        admins = store.query_nodes("Agent", filters={"role": "admin"})
        assert len(admins) == 1
        assert admins[0].properties["name"] == "alice"

    def test_query_nodes_respects_limit(self, store: InMemoryGraphStore) -> None:
        for i in range(10):
            store.add_node("Agent", {"name": f"agent-{i}"})
        results = store.query_nodes("Agent", limit=3)
        assert len(results) == 3

    def test_query_nodes_empty_for_unknown_type(self, store: InMemoryGraphStore) -> None:
        results = store.query_nodes("UnknownType")
        assert results == []

    def test_update_node_properties(self, store: InMemoryGraphStore) -> None:
        store.add_node("Agent", {"name": "eve", "status": "active"}, node_id="e1")
        result = store.update_node("e1", {"status": "inactive", "level": "5"})
        assert result is True
        node = store.get_node("e1")
        assert node is not None
        assert node.properties["status"] == "inactive"
        assert node.properties["level"] == "5"
        assert node.properties["name"] == "eve"  # Unchanged

    def test_update_nonexistent_node_returns_false(self, store: InMemoryGraphStore) -> None:
        result = store.update_node("ghost", {"name": "phantom"})
        assert result is False

    def test_delete_node(self, store: InMemoryGraphStore) -> None:
        store.add_node("Agent", {"name": "frank"}, node_id="f1")
        assert store.delete_node("f1") is True
        assert store.get_node("f1") is None

    def test_delete_nonexistent_node_returns_false(self, store: InMemoryGraphStore) -> None:
        assert store.delete_node("ghost") is False

    def test_delete_node_removes_connected_edges(self, store: InMemoryGraphStore) -> None:
        store.add_node("Agent", {"name": "a"}, node_id="a1")
        store.add_node("Agent", {"name": "b"}, node_id="b1")
        store.add_edge("a1", "b1", "KNOWS")
        store.delete_node("a1")
        # Neighbor query from b1 should return nothing.
        neighbors = store.query_neighbors("b1")
        assert len(neighbors) == 0


# ── Search ──────────────────────────────────────────────────────────


class TestSearch:
    """Full-text keyword search via search_nodes."""

    def test_search_finds_matching_nodes(self, store: InMemoryGraphStore) -> None:
        store.add_node("Fact", {"content": "Python is great"})
        store.add_node("Fact", {"content": "Rust is fast"})
        store.add_node("Fact", {"content": "Python typing is useful"})
        results = store.search_nodes("Fact", ["content"], "python")
        assert len(results) == 2

    def test_search_case_insensitive(self, store: InMemoryGraphStore) -> None:
        store.add_node("Fact", {"content": "POSTGRESQL is powerful"})
        results = store.search_nodes("Fact", ["content"], "postgresql")
        assert len(results) == 1

    def test_search_multiple_fields(self, store: InMemoryGraphStore) -> None:
        store.add_node("Fact", {"title": "Guide", "content": "Python basics"})
        store.add_node("Fact", {"title": "Python Advanced", "content": "Decorators"})
        results = store.search_nodes("Fact", ["title", "content"], "python")
        assert len(results) == 2

    def test_search_with_filters(self, store: InMemoryGraphStore) -> None:
        store.add_node("Fact", {"content": "Python guide", "source": "docs"})
        store.add_node("Fact", {"content": "Python tutorial", "source": "blog"})
        results = store.search_nodes(
            "Fact", ["content"], "python", filters={"source": "docs"}
        )
        assert len(results) == 1
        assert results[0].properties["source"] == "docs"

    def test_search_deduplication(self, store: InMemoryGraphStore) -> None:
        """A node matching multiple text_fields should appear only once."""
        store.add_node(
            "Fact",
            {"title": "Python tips", "content": "Python best practices"},
            node_id="dup-1",
        )
        results = store.search_nodes("Fact", ["title", "content"], "python")
        ids = [r.node_id for r in results]
        assert len(ids) == len(set(ids)), "Duplicate node IDs in search results"

    def test_search_respects_limit(self, store: InMemoryGraphStore) -> None:
        for i in range(20):
            store.add_node("Fact", {"content": f"Python fact {i}"})
        results = store.search_nodes("Fact", ["content"], "python", limit=5)
        assert len(results) == 5

    def test_search_no_matches(self, store: InMemoryGraphStore) -> None:
        store.add_node("Fact", {"content": "Rust programming"})
        results = store.search_nodes("Fact", ["content"], "haskell")
        assert results == []


# ── Edge operations ─────────────────────────────────────────────────


class TestEdgeOperations:
    """Edge add, query_neighbors, delete."""

    def test_add_edge_returns_graph_edge(self, store: InMemoryGraphStore) -> None:
        store.add_node("Agent", {"name": "a"}, node_id="a1")
        store.add_node("Agent", {"name": "b"}, node_id="b1")
        edge = store.add_edge("a1", "b1", "KNOWS", {"since": "2024"})
        assert isinstance(edge, GraphEdge)
        assert edge.source_id == "a1"
        assert edge.target_id == "b1"
        assert edge.edge_type == "KNOWS"
        assert edge.properties["since"] == "2024"

    def test_add_edge_missing_source_raises_key_error(
        self, store: InMemoryGraphStore
    ) -> None:
        store.add_node("Agent", {"name": "b"}, node_id="b1")
        with pytest.raises(KeyError, match="Source node not found"):
            store.add_edge("ghost", "b1", "KNOWS")

    def test_add_edge_missing_target_raises_key_error(
        self, store: InMemoryGraphStore
    ) -> None:
        store.add_node("Agent", {"name": "a"}, node_id="a1")
        with pytest.raises(KeyError, match="Target node not found"):
            store.add_edge("a1", "ghost", "KNOWS")

    def test_query_neighbors_outgoing(self, store: InMemoryGraphStore) -> None:
        store.add_node("Agent", {"name": "a"}, node_id="a1")
        store.add_node("Agent", {"name": "b"}, node_id="b1")
        store.add_edge("a1", "b1", "KNOWS")
        neighbors = store.query_neighbors("a1", direction=Direction.OUTGOING)
        assert len(neighbors) == 1
        edge, node = neighbors[0]
        assert node.node_id == "b1"

    def test_query_neighbors_incoming(self, store: InMemoryGraphStore) -> None:
        store.add_node("Agent", {"name": "a"}, node_id="a1")
        store.add_node("Agent", {"name": "b"}, node_id="b1")
        store.add_edge("a1", "b1", "KNOWS")
        neighbors = store.query_neighbors("b1", direction=Direction.INCOMING)
        assert len(neighbors) == 1
        edge, node = neighbors[0]
        assert node.node_id == "a1"

    def test_query_neighbors_both_directions(self, store: InMemoryGraphStore) -> None:
        store.add_node("Agent", {"name": "a"}, node_id="a1")
        store.add_node("Agent", {"name": "b"}, node_id="b1")
        store.add_edge("a1", "b1", "KNOWS")
        neighbors = store.query_neighbors("a1", direction=Direction.BOTH)
        assert len(neighbors) == 1

    def test_query_neighbors_with_edge_type_filter(
        self, store: InMemoryGraphStore
    ) -> None:
        store.add_node("Agent", {"name": "a"}, node_id="a1")
        store.add_node("Agent", {"name": "b"}, node_id="b1")
        store.add_node("Agent", {"name": "c"}, node_id="c1")
        store.add_edge("a1", "b1", "KNOWS")
        store.add_edge("a1", "c1", "WORKS_WITH")
        knows = store.query_neighbors("a1", edge_type="KNOWS")
        assert len(knows) == 1
        assert knows[0][1].node_id == "b1"

    def test_query_neighbors_missing_node(self, store: InMemoryGraphStore) -> None:
        assert store.query_neighbors("ghost") == []

    def test_delete_edge(self, store: InMemoryGraphStore) -> None:
        store.add_node("Agent", {"name": "a"}, node_id="a1")
        store.add_node("Agent", {"name": "b"}, node_id="b1")
        store.add_edge("a1", "b1", "KNOWS")
        result = store.delete_edge("a1", "b1", "KNOWS")
        assert result is True
        neighbors = store.query_neighbors("a1")
        assert len(neighbors) == 0

    def test_delete_nonexistent_edge_returns_false(
        self, store: InMemoryGraphStore
    ) -> None:
        store.add_node("Agent", {"name": "a"}, node_id="a1")
        store.add_node("Agent", {"name": "b"}, node_id="b1")
        result = store.delete_edge("a1", "b1", "GHOST_EDGE")
        assert result is False


# ── Traversal ───────────────────────────────────────────────────────


class TestTraversal:
    """BFS traversal tests."""

    def test_traverse_1_hop(self, store: InMemoryGraphStore) -> None:
        store.add_node("Agent", {"name": "a"}, node_id="a1")
        store.add_node("Fact", {"content": "f1"}, node_id="f1")
        store.add_edge("a1", "f1", "LEARNED")
        result = store.traverse("a1", max_hops=1)
        assert isinstance(result, TraversalResult)
        assert len(result.nodes) == 2  # a1 + f1
        assert len(result.edges) == 1
        assert len(result.paths) == 1

    def test_traverse_2_hops(self, store: InMemoryGraphStore) -> None:
        store.add_node("Agent", {"name": "a"}, node_id="a1")
        store.add_node("Fact", {"content": "f1"}, node_id="f1")
        store.add_node("Fact", {"content": "f2"}, node_id="f2")
        store.add_edge("a1", "f1", "LEARNED")
        store.add_edge("f1", "f2", "RELATES_TO")
        result = store.traverse("a1", max_hops=2)
        assert len(result.nodes) == 3
        assert len(result.paths) == 2

    def test_traverse_respects_max_hops(self, store: InMemoryGraphStore) -> None:
        store.add_node("Agent", {}, node_id="n0")
        store.add_node("Agent", {}, node_id="n1")
        store.add_node("Agent", {}, node_id="n2")
        store.add_node("Agent", {}, node_id="n3")
        store.add_edge("n0", "n1", "NEXT")
        store.add_edge("n1", "n2", "NEXT")
        store.add_edge("n2", "n3", "NEXT")
        result = store.traverse("n0", max_hops=2)
        visited_ids = {n.node_id for n in result.nodes}
        assert "n0" in visited_ids
        assert "n1" in visited_ids
        assert "n2" in visited_ids
        assert "n3" not in visited_ids  # 3 hops away, beyond limit

    def test_traverse_with_edge_type_filter(self, store: InMemoryGraphStore) -> None:
        store.add_node("Agent", {}, node_id="a1")
        store.add_node("Agent", {}, node_id="b1")
        store.add_node("Agent", {}, node_id="c1")
        store.add_edge("a1", "b1", "KNOWS")
        store.add_edge("a1", "c1", "BLOCKS")
        result = store.traverse("a1", edge_types=["KNOWS"], max_hops=1)
        visited_ids = {n.node_id for n in result.nodes}
        assert "b1" in visited_ids
        assert "c1" not in visited_ids

    def test_traverse_with_node_filter(self, store: InMemoryGraphStore) -> None:
        store.add_node("Agent", {"role": "admin"}, node_id="a1")
        store.add_node("Agent", {"role": "admin"}, node_id="b1")
        store.add_node("Agent", {"role": "user"}, node_id="c1")
        store.add_edge("a1", "b1", "KNOWS")
        store.add_edge("a1", "c1", "KNOWS")
        result = store.traverse("a1", max_hops=1, node_filter={"role": "admin"})
        visited_ids = {n.node_id for n in result.nodes}
        assert "b1" in visited_ids
        assert "c1" not in visited_ids

    def test_traverse_cross_boundary_detection(self) -> None:
        """Traversal across stores with different graph_origins sets crossed_boundaries."""
        store_a = InMemoryGraphStore(store_id="store-A")
        store_b = InMemoryGraphStore(store_id="store-B")

        # Add nodes from different origins into one store for traversal.
        store_a.add_node("Agent", {"name": "a"}, node_id="a1")
        # Manually inject a foreign-origin node to simulate hive merge.
        store_a._nodes["b1"] = {
            "type": "Agent",
            "properties": {"name": "b"},
            "graph_origin": "store-B",
        }
        store_a._edges.append({
            "edge_id": "e1",
            "source": "a1",
            "target": "b1",
            "type": "KNOWS",
            "properties": {},
            "graph_origin": "store-A",
        })

        result = store_a.traverse("a1", max_hops=1)
        assert result.crossed_boundaries is True

    def test_traverse_no_cross_boundary_same_origin(
        self, store: InMemoryGraphStore
    ) -> None:
        store.add_node("Agent", {}, node_id="a1")
        store.add_node("Agent", {}, node_id="b1")
        store.add_edge("a1", "b1", "KNOWS")
        result = store.traverse("a1", max_hops=1)
        assert result.crossed_boundaries is False

    def test_traverse_nonexistent_start(self, store: InMemoryGraphStore) -> None:
        result = store.traverse("ghost")
        assert result.nodes == []
        assert result.edges == []
        assert result.paths == []

    def test_traverse_incoming_direction(self, store: InMemoryGraphStore) -> None:
        store.add_node("Agent", {}, node_id="a1")
        store.add_node("Agent", {}, node_id="b1")
        store.add_edge("b1", "a1", "FOLLOWS")
        result = store.traverse("a1", max_hops=1, direction=Direction.INCOMING)
        visited_ids = {n.node_id for n in result.nodes}
        assert "b1" in visited_ids


# ── Thread safety ───────────────────────────────────────────────────


class TestThreadSafety:
    """Concurrent operations on InMemoryGraphStore."""

    def test_concurrent_add_nodes(self, store: InMemoryGraphStore) -> None:
        errors: list[str] = []

        def add_nodes(prefix: str) -> None:
            try:
                for i in range(50):
                    store.add_node("Agent", {"name": f"{prefix}-{i}"})
            except Exception as exc:
                errors.append(str(exc))

        threads = [
            threading.Thread(target=add_nodes, args=(f"t{t}",))
            for t in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
        agents = store.query_nodes("Agent", limit=300)
        assert len(agents) == 200  # 4 threads x 50 nodes


# ── Factory ─────────────────────────────────────────────────────────


class TestFactory:
    """create_hive_store factory function."""

    def test_create_memory_store(self) -> None:
        store = create_hive_store("memory", store_id="test-mem")
        assert isinstance(store, InMemoryGraphStore)
        assert store.store_id == "test-mem"

    def test_create_memory_store_default_id(self) -> None:
        store = create_hive_store("memory")
        assert store.store_id == "memory"

    def test_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown backend"):
            create_hive_store("neo4j")

    def test_postgres_backend_without_psycopg2_raises(self) -> None:
        """postgres+age should fail gracefully when psycopg2 is not installed."""
        # This test verifies the error path.  If psycopg2 IS installed,
        # it will fail on connection (no server), which is also acceptable.
        with pytest.raises((ImportError, Exception)):
            create_hive_store(
                "postgres+age",
                connection_string="host=localhost dbname=nonexistent",
            )


# ── Lifecycle ───────────────────────────────────────────────────────


class TestLifecycle:
    """Store lifecycle operations."""

    def test_close_clears_state(self, store: InMemoryGraphStore) -> None:
        store.add_node("Agent", {"name": "a"}, node_id="a1")
        store.close()
        assert store.get_node("a1") is None

    def test_close_is_idempotent(self, store: InMemoryGraphStore) -> None:
        store.close()
        store.close()  # Should not raise
