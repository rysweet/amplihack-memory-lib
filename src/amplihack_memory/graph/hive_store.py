"""HiveGraphStore -- Kuzu-backed graph store for the hive mind collective.

Extends KuzuGraphStore with hive-specific schema: agent registry nodes,
cross-agent confirmation edges, contradiction edges, and semantic bridges.

Public API:
    HiveGraphStore: KuzuGraphStore subclass with hive-specific operations.
"""

from __future__ import annotations

import time
from typing import Any

from .kuzu_store import KuzuGraphStore
from .types import GraphNode


class HiveGraphStore(KuzuGraphStore):
    """The hive mind's own graph -- metadata about the agent collective.

    Adds hive-specific tables on top of the base KuzuGraphStore:
    - HiveAgent nodes: registry with trust, domain, fact_count
    - CONFIRMED_BY edges: cross-agent fact confirmation
    - CONTRADICTS edges: detected conflicts between agents
    - SAME_CONCEPT edges: semantic bridges across agents

    Args:
        db_path: Filesystem path for the Kuzu database directory.
    """

    # Column schemas for hive-specific tables
    _HIVE_AGENT_COLUMNS = {
        "domain": "STRING",
        "trust": "STRING",
        "fact_count": "STRING",
        "joined_at": "STRING",
        "status": "STRING",
    }

    _CONFIRMED_BY_COLUMNS = {
        "confirmed_at": "STRING",
        "confidence": "STRING",
    }

    _CONTRADICTS_COLUMNS = {
        "detected_at": "STRING",
        "resolved": "STRING",
        "description": "STRING",
    }

    _SAME_CONCEPT_COLUMNS = {
        "similarity": "STRING",
        "detected_at": "STRING",
    }

    def __init__(self, db_path: str, buffer_pool_size: int = 256 * 1024 * 1024) -> None:
        super().__init__(db_path=db_path, store_id="__hive__", buffer_pool_size=buffer_pool_size)
        self._setup_hive_schema()

    def _setup_hive_schema(self) -> None:
        """Create hive-specific node and relationship tables."""
        # Node table for agent registry
        self.ensure_node_table("HiveAgent", self._HIVE_AGENT_COLUMNS)

        # Edge tables for cross-agent relationships.
        # These connect HiveAgent nodes to each other.
        self.ensure_rel_table(
            "CONFIRMED_BY", "HiveAgent", "HiveAgent",
            self._CONFIRMED_BY_COLUMNS,
        )
        self.ensure_rel_table(
            "CONTRADICTS", "HiveAgent", "HiveAgent",
            self._CONTRADICTS_COLUMNS,
        )
        self.ensure_rel_table(
            "SAME_CONCEPT", "HiveAgent", "HiveAgent",
            self._SAME_CONCEPT_COLUMNS,
        )

    def register_agent(
        self,
        agent_id: str,
        domain: str,
        trust: float = 1.0,
    ) -> GraphNode:
        """Register an agent in the hive graph.

        Creates a HiveAgent node with metadata about the agent's domain,
        initial trust score, and registration timestamp.

        Args:
            agent_id: Unique identifier for the agent.
            domain: Agent's domain of expertise.
            trust: Initial trust score (default 1.0).

        Returns:
            The created HiveAgent GraphNode.
        """
        return self.add_node(
            "HiveAgent",
            {
                "domain": domain,
                "trust": str(trust),
                "fact_count": "0",
                "joined_at": str(time.time()),
                "status": "active",
            },
            node_id=agent_id,
        )

    def get_agent_trust(self, agent_id: str) -> float:
        """Get the trust score for a registered agent.

        Args:
            agent_id: Agent to look up.

        Returns:
            Trust score as a float. Returns 0.0 if agent not found.
        """
        node = self.get_node(agent_id)
        if node is None:
            return 0.0
        try:
            return float(node.properties.get("trust", "0.0"))
        except (ValueError, TypeError):
            return 0.0

    def update_trust(self, agent_id: str, new_trust: float) -> None:
        """Update the trust score for a registered agent.

        Args:
            agent_id: Agent to update.
            new_trust: New trust score.

        Raises:
            KeyError: If agent is not registered.
        """
        node = self.get_node(agent_id)
        if node is None:
            raise KeyError(f"Agent not registered in hive: {agent_id}")
        self.update_node(agent_id, {"trust": str(new_trust)})

    def increment_fact_count(self, agent_id: str) -> None:
        """Increment the fact count for an agent.

        Args:
            agent_id: Agent whose count to increment.
        """
        node = self.get_node(agent_id)
        if node is None:
            return
        try:
            current = int(node.properties.get("fact_count", "0"))
        except (ValueError, TypeError):
            current = 0
        self.update_node(agent_id, {"fact_count": str(current + 1)})

    def get_expert_agents(self, domain: str) -> list[GraphNode]:
        """Find agents registered as experts in a given domain.

        Uses CONTAINS matching on the domain field, so partial matches
        work (e.g. "bio" matches "biology").

        Args:
            domain: Domain keyword to search for.

        Returns:
            List of HiveAgent nodes matching the domain, sorted by trust
            (highest first).
        """
        results = self.search_nodes(
            "HiveAgent",
            text_fields=["domain"],
            query=domain,
            limit=50,
        )
        # Sort by trust descending
        def _trust_key(n: GraphNode) -> float:
            try:
                return float(n.properties.get("trust", "0.0"))
            except (ValueError, TypeError):
                return 0.0

        results.sort(key=_trust_key, reverse=True)
        return results

    def add_confirmation(
        self,
        confirming_agent_id: str,
        confirmed_agent_id: str,
        confidence: float = 1.0,
    ) -> None:
        """Record that one agent confirms another agent's fact.

        Args:
            confirming_agent_id: Agent doing the confirming.
            confirmed_agent_id: Agent whose fact is confirmed.
            confidence: Confidence of the confirmation.
        """
        self.add_edge(
            confirming_agent_id,
            confirmed_agent_id,
            "CONFIRMED_BY",
            {
                "confirmed_at": str(time.time()),
                "confidence": str(confidence),
            },
        )

    def add_contradiction(
        self,
        agent_a_id: str,
        agent_b_id: str,
        description: str = "",
    ) -> None:
        """Record a detected contradiction between two agents.

        Args:
            agent_a_id: First agent in the contradiction.
            agent_b_id: Second agent in the contradiction.
            description: Description of the contradiction.
        """
        self.add_edge(
            agent_a_id,
            agent_b_id,
            "CONTRADICTS",
            {
                "detected_at": str(time.time()),
                "resolved": "false",
                "description": description,
            },
        )

    def add_semantic_bridge(
        self,
        agent_a_id: str,
        agent_b_id: str,
        similarity: float = 1.0,
    ) -> None:
        """Record a semantic bridge between two agents who share concepts.

        Args:
            agent_a_id: First agent.
            agent_b_id: Second agent.
            similarity: Semantic similarity score.
        """
        self.add_edge(
            agent_a_id,
            agent_b_id,
            "SAME_CONCEPT",
            {
                "similarity": str(similarity),
                "detected_at": str(time.time()),
            },
        )

    def get_confirmation_count(self, agent_id: str) -> int:
        """Count how many times this agent's facts have been confirmed.

        Args:
            agent_id: Agent to check.

        Returns:
            Number of incoming CONFIRMED_BY edges.
        """
        from .types import Direction

        neighbors = self.query_neighbors(
            agent_id,
            edge_type="CONFIRMED_BY",
            direction=Direction.INCOMING,
        )
        return len(neighbors)


__all__ = ["HiveGraphStore"]
