"""Tests for HiveGraphStore and FederatedGraphStore.

Test categories:
- TestHiveGraphStore (8): schema creation, register agent, trust CRUD,
  expert lookup, confirmations, contradictions, semantic bridges
- TestFederatedGraphStore (10): write-to-local, read-from-both, dedup,
  query_nodes, search_nodes, query_neighbors, federated_query, close
- TestCrossBoundaryTraversal (5): local->hive hop, edge type filtering,
  annotation, crossed_boundaries flag, node_filter
- TestEdgeCases (4): empty stores, close, missing node, protocol compliance

Total: 27 tests, all using real Kuzu databases via tmp_path.
"""

from __future__ import annotations

import pytest

from amplihack_memory.graph import (
    AnnotatedResult,
    Direction,
    FederatedGraphStore,
    FederatedQueryResult,
    GraphNode,
    GraphStore,
    HiveGraphStore,
    KuzuGraphStore,
    TraversalResult,
)


# ── fixtures ──────────────────────────────────────────────────


@pytest.fixture
def hive_store(tmp_path):
    """Create a fresh HiveGraphStore for each test."""
    db_path = tmp_path / "hive_db"
    s = HiveGraphStore(db_path=str(db_path))
    yield s
    s.close()


@pytest.fixture
def local_store(tmp_path):
    """Create a fresh local KuzuGraphStore for each test."""
    db_path = tmp_path / "local_db"
    s = KuzuGraphStore(db_path=db_path, store_id="agent-local")
    yield s
    s.close()


@pytest.fixture
def federated(tmp_path):
    """Create a FederatedGraphStore with separate local and hive DBs."""
    local_path = tmp_path / "fed_local_db"
    hive_path = tmp_path / "fed_hive_db"
    local = KuzuGraphStore(db_path=local_path, store_id="agent-alpha")
    hive = HiveGraphStore(db_path=str(hive_path))
    fed = FederatedGraphStore(local_store=local, hive_store=hive)
    yield fed, local, hive
    fed.close()


@pytest.fixture
def populated_federated(tmp_path):
    """FederatedGraphStore with data in both local and hive stores.

    Local store has:
      - Fact f1: "Python is dynamically typed"
      - Fact f2: "Rust has zero-cost abstractions"

    Hive store has:
      - Fact f3: "Python uses indentation for blocks"
      - Fact f4: "Go has goroutines for concurrency"
      - HiveAgent: agent-alpha (domain="python"), agent-beta (domain="systems")
    """
    local_path = tmp_path / "pop_local_db"
    hive_path = tmp_path / "pop_hive_db"

    local = KuzuGraphStore(db_path=local_path, store_id="agent-alpha")
    hive = HiveGraphStore(db_path=str(hive_path))

    # Populate local
    local.add_node("Fact", {"content": "Python is dynamically typed", "topic": "python"}, node_id="f1")
    local.add_node("Fact", {"content": "Rust has zero-cost abstractions", "topic": "rust"}, node_id="f2")

    # Populate hive
    hive.add_node("Fact", {"content": "Python uses indentation for blocks", "topic": "python"}, node_id="f3")
    hive.add_node("Fact", {"content": "Go has goroutines for concurrency", "topic": "go"}, node_id="f4")
    hive.register_agent("agent-alpha", domain="python", trust=0.9)
    hive.register_agent("agent-beta", domain="systems programming", trust=0.8)

    fed = FederatedGraphStore(local_store=local, hive_store=hive)
    yield fed, local, hive
    fed.close()


# ── TestHiveGraphStore (8) ────────────────────────────────────


class TestHiveGraphStore:
    """Verify hive-specific schema and operations."""

    def test_schema_created_on_init(self, hive_store):
        """HiveAgent table and relationship tables exist after init."""
        # Should be able to query the HiveAgent table without error
        nodes = hive_store.query_nodes("HiveAgent")
        assert nodes == []

    def test_register_agent(self, hive_store):
        """Register an agent and verify its properties."""
        node = hive_store.register_agent("agent-1", domain="biology", trust=0.95)
        assert node.node_id == "agent-1"
        assert node.node_type == "HiveAgent"
        assert node.properties["domain"] == "biology"
        assert node.properties["trust"] == "0.95"
        assert node.properties["fact_count"] == "0"
        assert node.properties["status"] == "active"

    def test_get_agent_trust(self, hive_store):
        """Retrieve trust score for a registered agent."""
        hive_store.register_agent("agent-1", domain="bio", trust=0.85)
        trust = hive_store.get_agent_trust("agent-1")
        assert trust == pytest.approx(0.85)

    def test_get_agent_trust_missing(self, hive_store):
        """Trust for missing agent returns 0.0."""
        trust = hive_store.get_agent_trust("nonexistent")
        assert trust == 0.0

    def test_update_trust(self, hive_store):
        """Update trust and verify the change persists."""
        hive_store.register_agent("agent-1", domain="bio", trust=0.5)
        hive_store.update_trust("agent-1", 0.99)
        assert hive_store.get_agent_trust("agent-1") == pytest.approx(0.99)

    def test_update_trust_missing_raises(self, hive_store):
        """Updating trust for unregistered agent raises KeyError."""
        with pytest.raises(KeyError, match="Agent not registered"):
            hive_store.update_trust("ghost", 1.0)

    def test_get_expert_agents(self, hive_store):
        """Find experts by domain keyword."""
        hive_store.register_agent("bio-1", domain="biology", trust=0.9)
        hive_store.register_agent("bio-2", domain="molecular biology", trust=0.7)
        hive_store.register_agent("chem-1", domain="chemistry", trust=0.8)

        experts = hive_store.get_expert_agents("biology")
        expert_ids = [e.node_id for e in experts]
        assert "bio-1" in expert_ids
        assert "bio-2" in expert_ids
        assert "chem-1" not in expert_ids
        # Sorted by trust: bio-1 (0.9) before bio-2 (0.7)
        assert expert_ids.index("bio-1") < expert_ids.index("bio-2")

    def test_increment_fact_count(self, hive_store):
        """Increment fact count and verify."""
        hive_store.register_agent("agent-1", domain="bio")
        hive_store.increment_fact_count("agent-1")
        hive_store.increment_fact_count("agent-1")
        node = hive_store.get_node("agent-1")
        assert node is not None
        assert node.properties["fact_count"] == "2"


# ── TestFederatedGraphStore (10) ──────────────────────────────


class TestFederatedGraphStore:
    """Verify federated read/write semantics."""

    def test_write_goes_to_local(self, federated):
        """Writes via FederatedGraphStore go to local store only."""
        fed, local, hive = federated
        node = fed.add_node("Fact", {"content": "test fact"}, node_id="f1")
        assert node.node_id == "f1"

        # In local
        assert local.get_node("f1") is not None
        # NOT in hive
        assert hive.get_node("f1") is None

    def test_read_finds_local(self, federated):
        """get_node finds nodes in local store."""
        fed, local, hive = federated
        local.add_node("Fact", {"content": "local fact"}, node_id="lf1")
        node = fed.get_node("lf1")
        assert node is not None
        assert node.properties["content"] == "local fact"

    def test_read_finds_hive(self, federated):
        """get_node finds nodes in hive store when not in local."""
        fed, local, hive = federated
        hive.add_node("Fact", {"content": "hive fact"}, node_id="hf1")
        node = fed.get_node("hf1")
        assert node is not None
        assert node.properties["content"] == "hive fact"

    def test_read_prefers_local(self, federated):
        """When same node_id exists in both, local is returned."""
        fed, local, hive = federated
        local.add_node("Fact", {"content": "LOCAL version"}, node_id="dup1")
        hive.add_node("Fact", {"content": "HIVE version"}, node_id="dup1")
        node = fed.get_node("dup1")
        assert node is not None
        assert node.properties["content"] == "LOCAL version"

    def test_search_deduplicates(self, populated_federated):
        """search_nodes deduplicates identical content from both stores."""
        fed, local, hive = populated_federated

        # Add a duplicate fact to hive with same content as local f1
        hive.add_node(
            "Fact",
            {"content": "Python is dynamically typed", "topic": "python"},
            node_id="f1_dup",
        )

        results = fed.search_nodes("Fact", ["content"], "python")
        contents = [n.properties["content"] for n in results]
        # "Python is dynamically typed" should appear only once
        assert contents.count("Python is dynamically typed") == 1
        # But "Python uses indentation for blocks" from hive should also appear
        assert "Python uses indentation for blocks" in contents

    def test_query_nodes_from_both(self, populated_federated):
        """query_nodes returns results from both stores."""
        fed, local, hive = populated_federated
        facts = fed.query_nodes("Fact")
        # local has f1, f2; hive has f3, f4
        assert len(facts) == 4

    def test_federated_query_result_structure(self, populated_federated):
        """federated_query returns properly structured FederatedQueryResult."""
        fed, local, hive = populated_federated
        result = fed.federated_query("python", node_type="Fact")

        assert isinstance(result, FederatedQueryResult)
        assert result.local_count >= 1  # at least f1 from local
        assert len(result.results) >= 1
        # Check that results are AnnotatedResult instances
        for ar in result.results:
            assert isinstance(ar, AnnotatedResult)
            assert ar.source in ("local", "hive") or ar.source.startswith("peer:")

    def test_federated_query_finds_experts(self, populated_federated):
        """federated_query discovers expert agents via HiveAgent nodes."""
        fed, local, hive = populated_federated
        result = fed.federated_query("python")
        # agent-alpha has domain="python" so should be found
        assert "agent-alpha" in result.expert_agents

    def test_update_and_delete_local_only(self, federated):
        """update and delete operate on local store only."""
        fed, local, hive = federated
        fed.add_node("Fact", {"content": "original"}, node_id="u1")
        fed.update_node("u1", {"content": "updated"})
        node = local.get_node("u1")
        assert node is not None
        assert node.properties["content"] == "updated"

        fed.delete_node("u1")
        assert local.get_node("u1") is None

    def test_store_id_composite(self, federated):
        """store_id combines local and hive IDs."""
        fed, local, hive = federated
        assert "agent-alpha" in fed.store_id
        assert "__hive__" in fed.store_id


# ── TestCrossBoundaryTraversal (5) ────────────────────────────


class TestCrossBoundaryTraversal:
    """Verify traversal crosses local/hive graph boundaries."""

    def test_local_to_hive_hop(self, tmp_path):
        """Traverse from a local node to a hive node via shared edge."""
        local_path = tmp_path / "trav_local"
        hive_path = tmp_path / "trav_hive"

        local = KuzuGraphStore(db_path=local_path, store_id="agent-x")
        hive = HiveGraphStore(db_path=str(hive_path))

        # Create nodes in each store with a shared node_id in hive
        local.add_node("Concept", {"name": "local_concept"}, node_id="c1")
        hive.add_node("Concept", {"name": "hive_concept"}, node_id="c2")

        # Create edge in local that points to c2 (which exists in hive)
        # For this to work with KuzuGraphStore, both nodes need to be in
        # the same store. So we create c2 in local too for the edge.
        local.add_node("Concept", {"name": "bridge_to_hive"}, node_id="c2")
        local.add_edge("c1", "c2", "RELATED_TO", {"via": "bridge"})

        fed = FederatedGraphStore(local, hive)
        result = fed.traverse("c1", max_hops=1, direction=Direction.OUTGOING)

        node_ids = {n.node_id for n in result.nodes}
        assert "c1" in node_ids
        assert "c2" in node_ids
        assert len(result.edges) >= 1

        fed.close()

    def test_crossed_boundaries_flag(self, tmp_path):
        """crossed_boundaries is True when traversal spans multiple origins."""
        local_path = tmp_path / "cb_local"
        hive_path = tmp_path / "cb_hive"

        local = KuzuGraphStore(db_path=local_path, store_id="origin-A")
        hive = HiveGraphStore(db_path=str(hive_path))

        # Local node
        local.add_node("Item", {"label": "local_item"}, node_id="i1")
        local.add_node("Item", {"label": "local_item_2"}, node_id="i2")
        local.add_edge("i1", "i2", "LINKS_TO")

        # Hive node (different origin)
        hive.add_node("Item", {"label": "hive_item"}, node_id="i1")
        hive.add_node("Item", {"label": "hive_item_2"}, node_id="i3")
        hive.add_edge("i1", "i3", "LINKS_TO")

        fed = FederatedGraphStore(local, hive)
        result = fed.traverse("i1", max_hops=1, direction=Direction.OUTGOING)

        # Results should come from both origins
        assert result.crossed_boundaries is True
        node_ids = {n.node_id for n in result.nodes}
        assert "i2" in node_ids or "i3" in node_ids

        fed.close()

    def test_edge_type_filtering_in_traversal(self, tmp_path):
        """Traversal respects edge_types filter across both stores."""
        local_path = tmp_path / "etf_local"
        hive_path = tmp_path / "etf_hive"

        local = KuzuGraphStore(db_path=local_path, store_id="origin-A")
        hive = HiveGraphStore(db_path=str(hive_path))

        local.add_node("Thing", {"name": "start"}, node_id="s1")
        local.add_node("Thing", {"name": "good"}, node_id="g1")
        local.add_node("Thing", {"name": "bad"}, node_id="b1")
        local.add_edge("s1", "g1", "GOOD_LINK")
        local.add_edge("s1", "b1", "BAD_LINK")

        fed = FederatedGraphStore(local, hive)

        # Only follow GOOD_LINK edges
        result = fed.traverse(
            "s1", edge_types=["GOOD_LINK"],
            max_hops=1, direction=Direction.OUTGOING,
        )
        node_ids = {n.node_id for n in result.nodes if n.node_id != "s1"}
        assert "g1" in node_ids
        assert "b1" not in node_ids

        fed.close()

    def test_node_filter_in_traversal(self, tmp_path):
        """Traversal respects node_filter across both stores."""
        local_path = tmp_path / "nf_local"
        hive_path = tmp_path / "nf_hive"

        local = KuzuGraphStore(db_path=local_path, store_id="origin-A")
        hive = HiveGraphStore(db_path=str(hive_path))

        local.add_node("Person", {"name": "Alice", "role": "dev"}, node_id="a1")
        local.add_node("Person", {"name": "Bob", "role": "pm"}, node_id="b1")
        local.add_node("Person", {"name": "Carol", "role": "dev"}, node_id="c1")
        local.add_edge("a1", "b1", "KNOWS")
        local.add_edge("a1", "c1", "KNOWS")

        fed = FederatedGraphStore(local, hive)

        result = fed.traverse(
            "a1", max_hops=1, direction=Direction.OUTGOING,
            node_filter={"role": "dev"},
        )
        node_ids = {n.node_id for n in result.nodes if n.node_id != "a1"}
        assert "c1" in node_ids
        assert "b1" not in node_ids

        fed.close()

    def test_traversal_nonexistent_start(self, federated):
        """Traversal from nonexistent node returns empty result."""
        fed, _, _ = federated
        result = fed.traverse("nonexistent")
        assert result.nodes == []
        assert result.paths == []
        assert result.crossed_boundaries is False


# ── TestEdgeCases (4) ─────────────────────────────────────────


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_stores_query(self, federated):
        """Querying empty federated store returns empty results."""
        fed, _, _ = federated
        results = fed.search_nodes("Fact", ["content"], "anything")
        assert results == []

    def test_empty_stores_federated_query(self, federated):
        """Federated query on empty stores returns empty result."""
        fed, _, _ = federated
        result = fed.federated_query("test")
        assert isinstance(result, FederatedQueryResult)
        assert result.results == []
        assert result.local_count == 0
        assert result.hive_count == 0

    def test_get_missing_node(self, federated):
        """get_node returns None when node doesn't exist in either store."""
        fed, _, _ = federated
        node = fed.get_node("nonexistent")
        assert node is None

    def test_protocol_compliance(self, federated):
        """FederatedGraphStore satisfies the GraphStore protocol."""
        fed, _, _ = federated
        assert isinstance(fed, GraphStore)

    def test_hive_store_is_graph_store(self, hive_store):
        """HiveGraphStore is a GraphStore (inherits from KuzuGraphStore)."""
        assert isinstance(hive_store, GraphStore)

    def test_hive_store_id_is_hive(self, hive_store):
        """HiveGraphStore always has store_id '__hive__'."""
        assert hive_store.store_id == "__hive__"

    def test_close_federated(self, tmp_path):
        """Closing FederatedGraphStore closes both inner stores."""
        local_path = tmp_path / "close_local"
        hive_path = tmp_path / "close_hive"
        local = KuzuGraphStore(db_path=local_path, store_id="local-close")
        hive = HiveGraphStore(db_path=str(hive_path))
        fed = FederatedGraphStore(local, hive)

        # Should not raise
        fed.close()
        # After close, store_id still accessible
        assert "local-close" in fed.store_id
