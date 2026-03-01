"""PostgresGraphStore -- GraphStore backed by PostgreSQL + Apache AGE.

For distributed hive mind: multiple agents on different machines connect
to the same PostgreSQL database. Graph queries use openCypher via AGE.

Supports two modes:
- Real PostgreSQL (production): connection_string to Azure PG
- In-memory dict (testing): InMemoryGraphStore implements the same interface

Public API:
    PostgresGraphStore: Distributed GraphStore implementation.
    InMemoryGraphStore: Dict-based fallback for testing without PostgreSQL.
    create_hive_store: Factory function for creating hive graph stores.
"""

from __future__ import annotations

import json
import threading
import uuid
from collections import deque
from typing import Any

from .types import Direction, GraphEdge, GraphNode, TraversalResult


# ── InMemoryGraphStore ──────────────────────────────────────────────


class InMemoryGraphStore:
    """Dict-based GraphStore for testing without PostgreSQL.

    Implements the same interface as PostgresGraphStore using plain dicts.
    This lets tests run without any database.  Thread-safe via a
    reentrant lock.

    Args:
        store_id: Human-readable identifier for this store instance.
    """

    def __init__(self, store_id: str = "in_memory") -> None:
        self._store_id = store_id
        self._nodes: dict[str, dict[str, Any]] = {}  # node_id -> {type, properties, graph_origin}
        self._edges: list[dict[str, Any]] = []  # [{edge_id, source, target, type, properties, graph_origin}]
        self._lock = threading.RLock()

    @property
    def store_id(self) -> str:
        return self._store_id

    # ── node operations ──────────────────────────────────────

    def add_node(
        self,
        node_type: str,
        properties: dict[str, Any],
        node_id: str | None = None,
    ) -> GraphNode:
        nid = node_id or uuid.uuid4().hex
        with self._lock:
            self._nodes[nid] = {
                "type": node_type,
                "properties": dict(properties),
                "graph_origin": self._store_id,
            }
        return GraphNode(
            node_id=nid,
            node_type=node_type,
            properties=dict(properties),
            graph_origin=self._store_id,
        )

    def get_node(self, node_id: str) -> GraphNode | None:
        with self._lock:
            entry = self._nodes.get(node_id)
        if entry is None:
            return None
        return GraphNode(
            node_id=node_id,
            node_type=entry["type"],
            properties=dict(entry["properties"]),
            graph_origin=entry["graph_origin"],
        )

    def query_nodes(
        self,
        node_type: str,
        filters: dict[str, Any] | None = None,
        limit: int = 50,
    ) -> list[GraphNode]:
        results: list[GraphNode] = []
        with self._lock:
            for nid, entry in self._nodes.items():
                if entry["type"] != node_type:
                    continue
                if filters:
                    if not all(
                        str(entry["properties"].get(k)) == str(v)
                        for k, v in filters.items()
                    ):
                        continue
                results.append(
                    GraphNode(
                        node_id=nid,
                        node_type=entry["type"],
                        properties=dict(entry["properties"]),
                        graph_origin=entry["graph_origin"],
                    )
                )
                if len(results) >= limit:
                    break
        return results

    def search_nodes(
        self,
        node_type: str,
        text_fields: list[str],
        query: str,
        filters: dict[str, Any] | None = None,
        limit: int = 10,
    ) -> list[GraphNode]:
        q_lower = query.lower()
        results: list[GraphNode] = []
        seen_ids: set[str] = set()
        with self._lock:
            for nid, entry in self._nodes.items():
                if entry["type"] != node_type:
                    continue
                # Text search: any specified field contains the query.
                matched = False
                for field in text_fields:
                    val = entry["properties"].get(field, "")
                    if q_lower in str(val).lower():
                        matched = True
                        break
                if not matched:
                    continue
                # Extra equality filters.
                if filters:
                    if not all(
                        str(entry["properties"].get(k)) == str(v)
                        for k, v in filters.items()
                    ):
                        continue
                if nid not in seen_ids:
                    seen_ids.add(nid)
                    results.append(
                        GraphNode(
                            node_id=nid,
                            node_type=entry["type"],
                            properties=dict(entry["properties"]),
                            graph_origin=entry["graph_origin"],
                        )
                    )
                if len(results) >= limit:
                    break
        return results

    def update_node(self, node_id: str, properties: dict[str, Any]) -> bool:
        with self._lock:
            entry = self._nodes.get(node_id)
            if entry is None:
                return False
            entry["properties"].update(properties)
        return True

    def delete_node(self, node_id: str) -> bool:
        with self._lock:
            if node_id not in self._nodes:
                return False
            del self._nodes[node_id]
            # Remove edges referencing this node.
            self._edges = [
                e for e in self._edges
                if e["source"] != node_id and e["target"] != node_id
            ]
        return True

    # ── edge operations ──────────────────────────────────────

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        properties: dict[str, Any] | None = None,
    ) -> GraphEdge:
        with self._lock:
            if source_id not in self._nodes:
                raise KeyError(f"Source node not found: {source_id}")
            if target_id not in self._nodes:
                raise KeyError(f"Target node not found: {target_id}")
            eid = uuid.uuid4().hex
            edge_record = {
                "edge_id": eid,
                "source": source_id,
                "target": target_id,
                "type": edge_type,
                "properties": dict(properties or {}),
                "graph_origin": self._store_id,
            }
            self._edges.append(edge_record)
        return GraphEdge(
            edge_id=eid,
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            properties=dict(properties or {}),
            graph_origin=self._store_id,
        )

    def query_neighbors(
        self,
        node_id: str,
        edge_type: str | None = None,
        direction: Direction = Direction.BOTH,
        limit: int = 50,
    ) -> list[tuple[GraphEdge, GraphNode]]:
        with self._lock:
            if node_id not in self._nodes:
                return []
            results: list[tuple[GraphEdge, GraphNode]] = []
            for e in self._edges:
                if edge_type is not None and e["type"] != edge_type:
                    continue
                neighbor_id: str | None = None
                if direction in (Direction.OUTGOING, Direction.BOTH) and e["source"] == node_id:
                    neighbor_id = e["target"]
                elif direction in (Direction.INCOMING, Direction.BOTH) and e["target"] == node_id:
                    neighbor_id = e["source"]
                if neighbor_id is None:
                    continue
                neighbor_entry = self._nodes.get(neighbor_id)
                if neighbor_entry is None:
                    continue
                edge = GraphEdge(
                    edge_id=e["edge_id"],
                    source_id=e["source"],
                    target_id=e["target"],
                    edge_type=e["type"],
                    properties=dict(e["properties"]),
                    graph_origin=e["graph_origin"],
                )
                neighbor = GraphNode(
                    node_id=neighbor_id,
                    node_type=neighbor_entry["type"],
                    properties=dict(neighbor_entry["properties"]),
                    graph_origin=neighbor_entry["graph_origin"],
                )
                results.append((edge, neighbor))
                if len(results) >= limit:
                    break
        return results

    def delete_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
    ) -> bool:
        with self._lock:
            original_len = len(self._edges)
            self._edges = [
                e for e in self._edges
                if not (
                    e["source"] == source_id
                    and e["target"] == target_id
                    and e["type"] == edge_type
                )
            ]
            return len(self._edges) < original_len

    # ── traversal ────────────────────────────────────────────

    def traverse(
        self,
        start_id: str,
        edge_types: list[str] | None = None,
        max_hops: int = 3,
        direction: Direction = Direction.OUTGOING,
        node_filter: dict[str, Any] | None = None,
    ) -> TraversalResult:
        start_node = self.get_node(start_id)
        if start_node is None:
            return TraversalResult()

        visited_ids: set[str] = {start_id}
        all_nodes: dict[str, GraphNode] = {start_id: start_node}
        all_edges: list[GraphEdge] = []
        paths: list[list] = []
        origins: set[str] = {start_node.graph_origin}

        queue: deque[tuple[str, list, int]] = deque()
        queue.append((start_id, [start_node], 0))

        while queue:
            current_id, current_path, hops = queue.popleft()
            if hops >= max_hops:
                continue

            if edge_types:
                neighbors: list[tuple[GraphEdge, GraphNode]] = []
                for et in edge_types:
                    neighbors.extend(
                        self.query_neighbors(current_id, et, direction)
                    )
            else:
                neighbors = self.query_neighbors(current_id, direction=direction)

            for edge, neighbor in neighbors:
                if node_filter:
                    if not all(
                        str(neighbor.properties.get(k)) == str(v)
                        for k, v in node_filter.items()
                    ):
                        continue

                all_edges.append(edge)
                origins.add(neighbor.graph_origin)
                new_path = current_path + [edge, neighbor]

                if neighbor.node_id not in visited_ids:
                    visited_ids.add(neighbor.node_id)
                    all_nodes[neighbor.node_id] = neighbor
                    queue.append((neighbor.node_id, new_path, hops + 1))

                paths.append(new_path)

        return TraversalResult(
            paths=paths,
            nodes=list(all_nodes.values()),
            edges=all_edges,
            crossed_boundaries=len(origins) > 1,
        )

    # ── lifecycle ────────────────────────────────────────────

    def close(self) -> None:
        with self._lock:
            self._nodes.clear()
            self._edges.clear()


# ── PostgresGraphStore ──────────────────────────────────────────────


class PostgresGraphStore:
    """GraphStore backed by PostgreSQL with Apache AGE extension.

    Provides distributed graph storage for the hive mind.  Multiple agents
    on different machines can connect to the same PostgreSQL database and
    share a single graph via openCypher queries executed through AGE.

    Args:
        connection_string: PostgreSQL connection string (DSN or URI).
        graph_name: AGE graph name.  Defaults to ``"hive_mind"``.
        store_id: Human-readable identifier.  Defaults to ``"__hive__"``.
    """

    def __init__(
        self,
        connection_string: str,
        graph_name: str = "hive_mind",
        store_id: str = "__hive__",
    ) -> None:
        self._store_id = store_id
        self._graph_name = graph_name
        self._connection_string = connection_string
        self._lock = threading.RLock()

        # Lazy imports -- fail fast with a clear message if missing.
        try:
            import psycopg2
        except ImportError as exc:
            raise ImportError(
                "psycopg2 is required for PostgresGraphStore.  "
                "Install it with: pip install psycopg2-binary"
            ) from exc

        self._conn = psycopg2.connect(connection_string)
        self._conn.autocommit = True

        self._ensure_age_extension()
        self._ensure_graph()

        # Track known node labels so we can scan them for get_node.
        self._known_labels: set[str] = set()

    @property
    def store_id(self) -> str:
        return self._store_id

    # ── AGE bootstrap ────────────────────────────────────────

    def _execute_cypher(
        self,
        cypher: str,
        params: dict[str, Any] | None = None,
    ) -> list[list[Any]]:
        """Execute an openCypher query via AGE and return all result rows.

        AGE queries are wrapped in ``SELECT * FROM cypher(...)`` and must
        include a column definition list.  This helper builds that wrapper
        automatically.

        Parameters embedded in *params* are serialised to JSON for safe
        interpolation into the Cypher string (AGE does not support
        server-side parameter binding the way native psycopg2 does).
        """
        safe_cypher = cypher
        if params:
            for key, value in params.items():
                # AGE Cypher uses $param syntax but parameter binding goes
                # through SQL, so we substitute safely via JSON encoding.
                safe_cypher = safe_cypher.replace(
                    f"${key}", _age_literal(value)
                )

        # AGE requires SET search_path and explicit ag_catalog usage.
        # Use a custom dollar-quote tag to reduce collision risk.
        sql = (
            f"SELECT * FROM cypher('{self._graph_name}', $_hive_cypher$ "
            f"{safe_cypher} "
            f"$_hive_cypher$) AS (result agtype);"
        )

        with self._lock:
            cur = self._conn.cursor()
            try:
                cur.execute("SET search_path = ag_catalog, \"$user\", public;")
                cur.execute(sql)
                rows = cur.fetchall()
            finally:
                cur.close()

        return [list(row) for row in rows]

    def _ensure_age_extension(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            try:
                cur.execute("CREATE EXTENSION IF NOT EXISTS age;")
                cur.execute("SET search_path = ag_catalog, \"$user\", public;")
            finally:
                cur.close()

    def _ensure_graph(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            try:
                cur.execute("SET search_path = ag_catalog, \"$user\", public;")
                cur.execute(
                    "SELECT count(*) FROM ag_graph WHERE name = %s;",
                    (self._graph_name,),
                )
                row = cur.fetchone()
                if row is None or row[0] == 0:
                    cur.execute(
                        "SELECT create_graph(%s);", (self._graph_name,)
                    )
            finally:
                cur.close()

    # ── node operations ──────────────────────────────────────

    def add_node(
        self,
        node_type: str,
        properties: dict[str, Any],
        node_id: str | None = None,
    ) -> GraphNode:
        nid = node_id or uuid.uuid4().hex
        # Build property assignments using _age_literal for safe escaping
        # instead of raw json.dumps (which bypasses injection protection).
        prop_parts = [
            f"node_id: {_age_literal(nid)}",
            f"graph_origin: {_age_literal(self._store_id)}",
        ]
        for k, v in properties.items():
            prop_parts.append(f"{k}: {_age_literal(str(v))}")
        props_str = ", ".join(prop_parts)

        cypher = f"CREATE (n:{node_type} {{{props_str}}}) RETURN n"
        self._execute_cypher(cypher)

        self._known_labels.add(node_type)
        return GraphNode(
            node_id=nid,
            node_type=node_type,
            properties=dict(properties),
            graph_origin=self._store_id,
        )

    def get_node(self, node_id: str) -> GraphNode | None:
        for label in list(self._known_labels):
            node = self._get_node_from_label(node_id, label)
            if node is not None:
                return node
        # Fall back to a label-less match.
        cypher = f"MATCH (n) WHERE n.node_id = {_age_literal(node_id)} RETURN n"
        rows = self._execute_cypher(cypher)
        if rows:
            return self._agtype_to_node(rows[0][0])
        return None

    def _get_node_from_label(self, node_id: str, label: str) -> GraphNode | None:
        cypher = (
            f"MATCH (n:{label}) WHERE n.node_id = {_age_literal(node_id)} RETURN n"
        )
        rows = self._execute_cypher(cypher)
        if rows:
            return self._agtype_to_node(rows[0][0], label)
        return None

    def query_nodes(
        self,
        node_type: str,
        filters: dict[str, Any] | None = None,
        limit: int = 50,
    ) -> list[GraphNode]:
        where_parts: list[str] = []
        for k, v in (filters or {}).items():
            where_parts.append(f"n.{k} = {_age_literal(str(v))}")
        where_clause = f" WHERE {' AND '.join(where_parts)}" if where_parts else ""
        cypher = f"MATCH (n:{node_type}){where_clause} RETURN n LIMIT {limit}"
        rows = self._execute_cypher(cypher)
        self._known_labels.add(node_type)
        return [self._agtype_to_node(r[0], node_type) for r in rows]

    def search_nodes(
        self,
        node_type: str,
        text_fields: list[str],
        query: str,
        filters: dict[str, Any] | None = None,
        limit: int = 10,
    ) -> list[GraphNode]:
        # AGE does not have a native CONTAINS; use string_contains or
        # fall back to property comparison in application code.
        # For simplicity we fetch all nodes of the type and filter
        # client-side, respecting the limit.
        all_of_type = self.query_nodes(node_type, filters=filters, limit=1000)
        q_lower = query.lower()
        results: list[GraphNode] = []
        seen: set[str] = set()
        for node in all_of_type:
            for field in text_fields:
                val = node.properties.get(field, "")
                if q_lower in str(val).lower():
                    if node.node_id not in seen:
                        seen.add(node.node_id)
                        results.append(node)
                    break
            if len(results) >= limit:
                break
        return results

    def update_node(self, node_id: str, properties: dict[str, Any]) -> bool:
        node = self.get_node(node_id)
        if node is None:
            return False
        set_parts = [
            f"n.{k} = {_age_literal(str(v))}" for k, v in properties.items()
        ]
        if not set_parts:
            return True
        set_clause = ", ".join(set_parts)
        label = node.node_type
        cypher = (
            f"MATCH (n:{label}) WHERE n.node_id = {_age_literal(node_id)} "
            f"SET {set_clause} RETURN n"
        )
        self._execute_cypher(cypher)
        return True

    def delete_node(self, node_id: str) -> bool:
        node = self.get_node(node_id)
        if node is None:
            return False
        label = node.node_type
        cypher = (
            f"MATCH (n:{label}) WHERE n.node_id = {_age_literal(node_id)} "
            f"DETACH DELETE n"
        )
        self._execute_cypher(cypher)
        return True

    # ── edge operations ──────────────────────────────────────

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        properties: dict[str, Any] | None = None,
    ) -> GraphEdge:
        src = self.get_node(source_id)
        tgt = self.get_node(target_id)
        if src is None:
            raise KeyError(f"Source node not found: {source_id}")
        if tgt is None:
            raise KeyError(f"Target node not found: {target_id}")

        eid = uuid.uuid4().hex
        # Build property assignments using _age_literal for safe escaping
        # instead of raw json.dumps (which bypasses injection protection).
        prop_parts = [
            f"edge_id: {_age_literal(eid)}",
            f"graph_origin: {_age_literal(self._store_id)}",
        ]
        for k, v in (properties or {}).items():
            prop_parts.append(f"{k}: {_age_literal(str(v))}")
        props_str = ", ".join(prop_parts)

        cypher = (
            f"MATCH (a:{src.node_type}), (b:{tgt.node_type}) "
            f"WHERE a.node_id = {_age_literal(source_id)} "
            f"AND b.node_id = {_age_literal(target_id)} "
            f"CREATE (a)-[r:{edge_type} {{{props_str}}}]->(b) RETURN r"
        )
        self._execute_cypher(cypher)

        return GraphEdge(
            edge_id=eid,
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            properties=dict(properties or {}),
            graph_origin=self._store_id,
        )

    def query_neighbors(
        self,
        node_id: str,
        edge_type: str | None = None,
        direction: Direction = Direction.BOTH,
        limit: int = 50,
    ) -> list[tuple[GraphEdge, GraphNode]]:
        node = self.get_node(node_id)
        if node is None:
            return []

        results: list[tuple[GraphEdge, GraphNode]] = []

        edge_filter = f":{edge_type}" if edge_type else ""

        if direction in (Direction.OUTGOING, Direction.BOTH):
            cypher = (
                f"MATCH (a:{node.node_type})-[r{edge_filter}]->(b) "
                f"WHERE a.node_id = {_age_literal(node_id)} "
                f"RETURN r, b LIMIT {limit}"
            )
            rows = self._execute_cypher(cypher)
            for row in rows:
                edge = self._agtype_to_edge(row[0], node_id, "outgoing")
                neighbor = self._agtype_to_node(row[1] if len(row) > 1 else row[0])
                results.append((edge, neighbor))

        if direction in (Direction.INCOMING, Direction.BOTH):
            cypher = (
                f"MATCH (a)-[r{edge_filter}]->(b:{node.node_type}) "
                f"WHERE b.node_id = {_age_literal(node_id)} "
                f"RETURN r, a LIMIT {limit}"
            )
            rows = self._execute_cypher(cypher)
            for row in rows:
                edge = self._agtype_to_edge(row[0], node_id, "incoming")
                neighbor = self._agtype_to_node(row[1] if len(row) > 1 else row[0])
                results.append((edge, neighbor))

        return results[:limit]

    def delete_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
    ) -> bool:
        src = self.get_node(source_id)
        tgt = self.get_node(target_id)
        if src is None or tgt is None:
            return False
        cypher = (
            f"MATCH (a:{src.node_type})-[r:{edge_type}]->(b:{tgt.node_type}) "
            f"WHERE a.node_id = {_age_literal(source_id)} "
            f"AND b.node_id = {_age_literal(target_id)} "
            f"DELETE r"
        )
        self._execute_cypher(cypher)
        return True

    # ── traversal ────────────────────────────────────────────

    def traverse(
        self,
        start_id: str,
        edge_types: list[str] | None = None,
        max_hops: int = 3,
        direction: Direction = Direction.OUTGOING,
        node_filter: dict[str, Any] | None = None,
    ) -> TraversalResult:
        """BFS traversal from start_id up to max_hops hops.

        Uses iterative BFS via query_neighbors, matching the Kuzu
        implementation pattern.
        """
        start_node = self.get_node(start_id)
        if start_node is None:
            return TraversalResult()

        visited_ids: set[str] = {start_id}
        all_nodes: dict[str, GraphNode] = {start_id: start_node}
        all_edges: list[GraphEdge] = []
        paths: list[list] = []
        origins: set[str] = {start_node.graph_origin}

        queue: deque[tuple[str, list, int]] = deque()
        queue.append((start_id, [start_node], 0))

        while queue:
            current_id, current_path, hops = queue.popleft()
            if hops >= max_hops:
                continue

            if edge_types:
                neighbors: list[tuple[GraphEdge, GraphNode]] = []
                for et in edge_types:
                    neighbors.extend(
                        self.query_neighbors(current_id, et, direction)
                    )
            else:
                neighbors = self.query_neighbors(
                    current_id, direction=direction
                )

            for edge, neighbor in neighbors:
                if node_filter:
                    if not all(
                        str(neighbor.properties.get(k)) == str(v)
                        for k, v in node_filter.items()
                    ):
                        continue

                all_edges.append(edge)
                origins.add(neighbor.graph_origin)
                new_path = current_path + [edge, neighbor]

                if neighbor.node_id not in visited_ids:
                    visited_ids.add(neighbor.node_id)
                    all_nodes[neighbor.node_id] = neighbor
                    queue.append((neighbor.node_id, new_path, hops + 1))

                paths.append(new_path)

        return TraversalResult(
            paths=paths,
            nodes=list(all_nodes.values()),
            edges=all_edges,
            crossed_boundaries=len(origins) > 1,
        )

    # ── lifecycle ────────────────────────────────────────────

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None  # type: ignore[assignment]

    # ── private helpers ──────────────────────────────────────

    def _agtype_to_node(
        self, agtype_val: Any, label_hint: str | None = None
    ) -> GraphNode:
        """Convert an AGE agtype vertex result to a GraphNode."""
        if isinstance(agtype_val, str):
            data = json.loads(agtype_val.rstrip("::vertex"))
        elif isinstance(agtype_val, dict):
            data = agtype_val
        else:
            data = {}

        props = data.get("properties", data)
        node_id = str(props.pop("node_id", ""))
        graph_origin = str(props.pop("graph_origin", ""))
        label = label_hint or str(data.get("label", props.pop("label", "")))

        return GraphNode(
            node_id=node_id,
            node_type=label,
            properties=props,
            graph_origin=graph_origin,
        )

    def _agtype_to_edge(
        self, agtype_val: Any, anchor_id: str, direction: str
    ) -> GraphEdge:
        """Convert an AGE agtype edge result to a GraphEdge."""
        if isinstance(agtype_val, str):
            data = json.loads(agtype_val.rstrip("::edge"))
        elif isinstance(agtype_val, dict):
            data = agtype_val
        else:
            data = {}

        props = data.get("properties", data)
        eid = str(props.pop("edge_id", ""))
        graph_origin = str(props.pop("graph_origin", ""))
        label = str(data.get("label", props.pop("label", "")))

        if direction == "outgoing":
            source_id = anchor_id
            target_id = ""
        else:
            source_id = ""
            target_id = anchor_id

        return GraphEdge(
            edge_id=eid,
            source_id=source_id,
            target_id=target_id,
            edge_type=label,
            properties=props,
            graph_origin=graph_origin,
        )


# ── Factory ─────────────────────────────────────────────────────────


def create_hive_store(backend: str = "postgres+age", **kwargs: Any) -> Any:
    """Create a hive graph store.

    Args:
        backend: ``"postgres+age"`` (production), ``"kuzu"`` (local),
            ``"memory"`` (testing).
        **kwargs: Backend-specific configuration.

    Returns:
        A GraphStore implementation.

    Raises:
        ValueError: If *backend* is unrecognised.
        ImportError: If required dependencies are not installed.
    """
    if backend == "postgres+age":
        return PostgresGraphStore(
            connection_string=kwargs["connection_string"],
            graph_name=kwargs.get("graph_name", "hive_mind"),
            store_id=kwargs.get("store_id", "__hive__"),
        )
    elif backend == "kuzu":
        from .kuzu_store import KuzuGraphStore

        return KuzuGraphStore(
            db_path=kwargs["db_path"],
            store_id=kwargs.get("store_id"),
        )
    elif backend == "memory":
        return InMemoryGraphStore(
            store_id=kwargs.get("store_id", "memory"),
        )
    else:
        raise ValueError(
            f"Unknown backend: {backend!r}.  "
            f"Choose from: 'postgres+age', 'kuzu', 'memory'"
        )


# ── Helpers ─────────────────────────────────────────────────────────


def _age_literal(value: Any) -> str:
    """Escape a Python value for safe interpolation into an AGE Cypher string.

    AGE does not support native parameter binding like psycopg2 ``%s``
    placeholders, so we JSON-encode values and wrap strings in single
    quotes for the Cypher layer.

    Values containing ``$$`` (or the custom dollar-quote tag) are rejected
    to prevent escaping out of the SQL dollar-quoted block.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    elif isinstance(value, (int, float)):
        return str(value)
    elif value is None:
        return "null"
    else:
        text = str(value)
        # Reject values that could break out of the dollar-quoted SQL block.
        if "$$" in text or "$_hive_cypher$" in text:
            raise ValueError(
                f"Value contains dollar-quote sequence and cannot be safely "
                f"interpolated into AGE Cypher: {text!r}"
            )
        # Escape single quotes for Cypher string literals.
        escaped = text.replace("\\", "\\\\").replace("'", "\\'")
        return f"'{escaped}'"


__all__ = [
    "PostgresGraphStore",
    "InMemoryGraphStore",
    "create_hive_store",
]
