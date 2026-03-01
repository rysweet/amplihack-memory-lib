"""Tests for the graph abstraction layer (types, protocol, KuzuGraphStore).

Test categories:
- TestGraphDataTypes: GraphNode, GraphEdge, TraversalResult dataclasses (5 tests)
- TestKuzuNodeCRUD: add, get, query, search, update, delete, filters, empty (8 tests)
- TestKuzuEdgeOperations: add, query_neighbors in/out/both, delete, missing node (6 tests)
- TestKuzuTraversal: 1-hop, 2-hop, 3-hop, edge type filter, node filter (5 tests)
- TestKuzuSchema: ensure_node_table idempotent, ensure_rel_table, multi-table (3 tests)
- TestProtocolCompliance: isinstance check, all methods present (2 tests)
- TestEdgeCases: empty store, close, get nonexistent (3 tests)

Total: 32 tests, all using real Kuzu databases via tmp_path.
"""

from __future__ import annotations

import pytest

from amplihack_memory.graph import (
    Direction,
    GraphEdge,
    GraphNode,
    GraphStore,
    KuzuGraphStore,
    TraversalResult,
)


# ── fixtures ──────────────────────────────────────────────────


@pytest.fixture
def store(tmp_path):
    """Create a fresh KuzuGraphStore for each test."""
    db_path = tmp_path / "test_graph_db"
    s = KuzuGraphStore(db_path=db_path, store_id="test-store")
    yield s
    s.close()


@pytest.fixture
def populated_store(store):
    """Store pre-populated with a small graph for traversal tests.

    Graph structure:
        alice --KNOWS--> bob --KNOWS--> carol --KNOWS--> dave
        alice --WORKS_WITH--> carol
    """
    alice = store.add_node("Person", {"name": "Alice", "role": "lead"}, node_id="alice")
    bob = store.add_node("Person", {"name": "Bob", "role": "dev"}, node_id="bob")
    carol = store.add_node("Person", {"name": "Carol", "role": "dev"}, node_id="carol")
    dave = store.add_node("Person", {"name": "Dave", "role": "pm"}, node_id="dave")

    store.add_edge("alice", "bob", "KNOWS", {"since": "2020"})
    store.add_edge("bob", "carol", "KNOWS", {"since": "2021"})
    store.add_edge("carol", "dave", "KNOWS", {"since": "2022"})
    store.add_edge("alice", "carol", "WORKS_WITH", {"project": "alpha"})

    return store


# ── TestGraphDataTypes (5) ────────────────────────────────────


class TestGraphDataTypes:
    """Verify the immutable data structures behave correctly."""

    def test_graph_node_creation(self):
        node = GraphNode(node_id="n1", node_type="Agent", properties={"name": "a1"})
        assert node.node_id == "n1"
        assert node.node_type == "Agent"
        assert node.properties == {"name": "a1"}
        assert node.graph_origin == ""

    def test_graph_node_frozen(self):
        node = GraphNode(node_id="n1", node_type="Agent")
        with pytest.raises(AttributeError):
            node.node_id = "n2"  # type: ignore[misc]

    def test_graph_edge_defaults(self):
        edge = GraphEdge()
        assert edge.edge_id == ""
        assert edge.source_id == ""
        assert edge.target_id == ""
        assert edge.edge_type == ""
        assert edge.properties == {}
        assert edge.graph_origin == ""

    def test_graph_edge_with_values(self):
        edge = GraphEdge(
            edge_id="e1",
            source_id="a",
            target_id="b",
            edge_type="KNOWS",
            properties={"weight": "0.9"},
            graph_origin="store-1",
        )
        assert edge.edge_id == "e1"
        assert edge.source_id == "a"
        assert edge.edge_type == "KNOWS"

    def test_traversal_result_defaults(self):
        tr = TraversalResult()
        assert tr.paths == []
        assert tr.nodes == []
        assert tr.edges == []
        assert tr.crossed_boundaries is False


# ── TestKuzuNodeCRUD (8) ──────────────────────────────────────


class TestKuzuNodeCRUD:
    """Node create/read/update/delete operations."""

    def test_add_node_auto_id(self, store):
        node = store.add_node("Agent", {"name": "agent-1"})
        assert node.node_id  # non-empty
        assert node.node_type == "Agent"
        assert node.properties["name"] == "agent-1"
        assert node.graph_origin == "test-store"

    def test_add_node_explicit_id(self, store):
        node = store.add_node("Agent", {"name": "agent-2"}, node_id="my-id")
        assert node.node_id == "my-id"

    def test_get_node(self, store):
        created = store.add_node("Agent", {"name": "getter"}, node_id="g1")
        fetched = store.get_node("g1")
        assert fetched is not None
        assert fetched.node_id == "g1"
        assert fetched.node_type == "Agent"
        assert fetched.properties["name"] == "getter"

    def test_query_nodes_all(self, store):
        store.add_node("Fact", {"text": "sky is blue"}, node_id="f1")
        store.add_node("Fact", {"text": "grass is green"}, node_id="f2")
        store.add_node("Agent", {"name": "a1"}, node_id="a1")

        facts = store.query_nodes("Fact")
        assert len(facts) == 2
        assert all(n.node_type == "Fact" for n in facts)

    def test_query_nodes_with_filter(self, store):
        store.add_node("Person", {"name": "Alice", "role": "dev"}, node_id="p1")
        store.add_node("Person", {"name": "Bob", "role": "pm"}, node_id="p2")
        store.add_node("Person", {"name": "Carol", "role": "dev"}, node_id="p3")

        devs = store.query_nodes("Person", filters={"role": "dev"})
        assert len(devs) == 2
        names = {n.properties["name"] for n in devs}
        assert names == {"Alice", "Carol"}

    def test_search_nodes(self, store):
        store.add_node("Fact", {"text": "Python is great", "topic": "lang"}, node_id="f1")
        store.add_node("Fact", {"text": "Rust is fast", "topic": "lang"}, node_id="f2")
        store.add_node("Fact", {"text": "SQL databases are useful", "topic": "db"}, node_id="f3")

        results = store.search_nodes("Fact", ["text"], "python")
        assert len(results) == 1
        assert results[0].properties["text"] == "Python is great"

    def test_update_node(self, store):
        store.add_node("Agent", {"name": "old-name", "status": "active"}, node_id="u1")
        success = store.update_node("u1", {"name": "new-name"})
        assert success is True

        updated = store.get_node("u1")
        assert updated is not None
        assert updated.properties["name"] == "new-name"

    def test_delete_node(self, store):
        store.add_node("Agent", {"name": "doomed"}, node_id="d1")
        assert store.get_node("d1") is not None

        success = store.delete_node("d1")
        assert success is True
        assert store.get_node("d1") is None


# ── TestKuzuEdgeOperations (6) ────────────────────────────────


class TestKuzuEdgeOperations:
    """Edge creation, neighbor queries, and deletion."""

    def test_add_edge(self, store):
        store.add_node("Person", {"name": "Alice"}, node_id="a")
        store.add_node("Person", {"name": "Bob"}, node_id="b")

        edge = store.add_edge("a", "b", "KNOWS", {"since": "2023"})
        assert edge.edge_type == "KNOWS"
        assert edge.properties["since"] == "2023"
        assert edge.graph_origin == "test-store"

    def test_add_edge_missing_source_raises(self, store):
        store.add_node("Person", {"name": "Bob"}, node_id="b")
        with pytest.raises(KeyError, match="Source node not found"):
            store.add_edge("nonexistent", "b", "KNOWS")

    def test_add_edge_missing_target_raises(self, store):
        store.add_node("Person", {"name": "Alice"}, node_id="a")
        with pytest.raises(KeyError, match="Target node not found"):
            store.add_edge("a", "nonexistent", "KNOWS")

    def test_query_neighbors_outgoing(self, populated_store):
        neighbors = populated_store.query_neighbors(
            "alice", edge_type="KNOWS", direction=Direction.OUTGOING,
        )
        assert len(neighbors) == 1
        edge, node = neighbors[0]
        assert edge.edge_type == "KNOWS"
        assert node.properties["name"] == "Bob"

    def test_query_neighbors_incoming(self, populated_store):
        neighbors = populated_store.query_neighbors(
            "bob", edge_type="KNOWS", direction=Direction.INCOMING,
        )
        assert len(neighbors) == 1
        _, node = neighbors[0]
        assert node.properties["name"] == "Alice"

    def test_delete_edge(self, store):
        store.add_node("Person", {"name": "A"}, node_id="a")
        store.add_node("Person", {"name": "B"}, node_id="b")
        store.add_edge("a", "b", "KNOWS")

        # Verify edge exists.
        neighbors = store.query_neighbors("a", "KNOWS", Direction.OUTGOING)
        assert len(neighbors) == 1

        success = store.delete_edge("a", "b", "KNOWS")
        assert success is True

        # Verify edge is gone.
        neighbors = store.query_neighbors("a", "KNOWS", Direction.OUTGOING)
        assert len(neighbors) == 0


# ── TestKuzuTraversal (5) ─────────────────────────────────────


class TestKuzuTraversal:
    """Multi-hop BFS traversal tests."""

    def test_traverse_1_hop(self, populated_store):
        result = populated_store.traverse("alice", max_hops=1, direction=Direction.OUTGOING)
        # alice -> bob (KNOWS), alice -> carol (WORKS_WITH)
        neighbor_ids = {n.node_id for n in result.nodes if n.node_id != "alice"}
        assert "bob" in neighbor_ids
        assert "carol" in neighbor_ids
        assert "dave" not in neighbor_ids

    def test_traverse_2_hops(self, populated_store):
        result = populated_store.traverse("alice", max_hops=2, direction=Direction.OUTGOING)
        node_ids = {n.node_id for n in result.nodes}
        # alice -> bob -> carol, alice -> carol -> dave
        assert "alice" in node_ids
        assert "bob" in node_ids
        assert "carol" in node_ids
        assert "dave" in node_ids

    def test_traverse_3_hops(self, populated_store):
        result = populated_store.traverse("alice", max_hops=3, direction=Direction.OUTGOING)
        node_ids = {n.node_id for n in result.nodes}
        assert "dave" in node_ids
        assert len(result.edges) > 0
        assert len(result.paths) > 0

    def test_traverse_edge_type_filter(self, populated_store):
        result = populated_store.traverse(
            "alice",
            edge_types=["KNOWS"],
            max_hops=3,
            direction=Direction.OUTGOING,
        )
        # Only KNOWS edges -- alice->bob->carol->dave
        node_ids = {n.node_id for n in result.nodes}
        assert "bob" in node_ids
        assert "carol" in node_ids
        assert "dave" in node_ids
        # All traversed edges should be KNOWS
        for edge in result.edges:
            assert edge.edge_type == "KNOWS"

    def test_traverse_node_filter(self, populated_store):
        result = populated_store.traverse(
            "alice",
            max_hops=3,
            direction=Direction.OUTGOING,
            node_filter={"role": "dev"},
        )
        # Only nodes with role=dev should be in the result
        for node in result.nodes:
            if node.node_id != "alice":
                assert node.properties.get("role") == "dev"


# ── TestKuzuSchema (3) ────────────────────────────────────────


class TestKuzuSchema:
    """Schema management and DDL idempotency."""

    def test_ensure_node_table_idempotent(self, store):
        store.ensure_node_table("Widget", {"color": "STRING", "size": "STRING"})
        store.ensure_node_table("Widget", {"color": "STRING", "size": "STRING"})
        # Should not raise. Add a node to verify table works.
        node = store.add_node("Widget", {"color": "red", "size": "large"})
        assert node.node_type == "Widget"

    def test_ensure_rel_table(self, store):
        store.ensure_node_table("A")
        store.ensure_node_table("B")
        store.ensure_rel_table("CONNECTS", "A", "B", {"weight": "STRING"})
        # Add nodes and edge to verify table works.
        store.add_node("A", {"name": "a1"}, node_id="a1")
        store.add_node("B", {"name": "b1"}, node_id="b1")
        edge = store.add_edge("a1", "b1", "CONNECTS", {"weight": "0.5"})
        assert edge.edge_type == "CONNECTS"

    def test_multi_table_search(self, store):
        store.add_node("Fact", {"text": "water is H2O"}, node_id="f1")
        store.add_node("Concept", {"text": "water cycle"}, node_id="c1")

        facts = store.search_nodes("Fact", ["text"], "water")
        concepts = store.search_nodes("Concept", ["text"], "water")

        assert len(facts) == 1
        assert len(concepts) == 1
        assert facts[0].node_type == "Fact"
        assert concepts[0].node_type == "Concept"


# ── TestProtocolCompliance (2) ────────────────────────────────


class TestProtocolCompliance:
    """Verify KuzuGraphStore satisfies the GraphStore protocol."""

    def test_isinstance_check(self, store):
        assert isinstance(store, GraphStore)

    def test_all_protocol_methods_present(self, store):
        required_methods = [
            "store_id", "add_node", "get_node", "query_nodes",
            "search_nodes", "update_node", "delete_node",
            "add_edge", "query_neighbors", "delete_edge",
            "traverse", "close",
        ]
        for method_name in required_methods:
            assert hasattr(store, method_name), f"Missing: {method_name}"


# ── TestEdgeCases (3) ─────────────────────────────────────────


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_store_query(self, store):
        # Querying a type that has never been registered returns empty.
        nodes = store.query_nodes("NonExistent")
        assert nodes == []

    def test_get_nonexistent_node(self, store):
        result = store.get_node("does-not-exist")
        assert result is None

    def test_delete_nonexistent_node(self, store):
        result = store.delete_node("does-not-exist")
        assert result is False

    def test_update_nonexistent_node(self, store):
        result = store.update_node("does-not-exist", {"key": "val"})
        assert result is False

    def test_traverse_nonexistent_start(self, store):
        result = store.traverse("does-not-exist")
        assert result.nodes == []
        assert result.paths == []

    def test_close_and_properties(self, store):
        # store_id should be accessible.
        assert store.store_id == "test-store"
        store.close()
        # After close, references are None but store_id still works.
        assert store.store_id == "test-store"
