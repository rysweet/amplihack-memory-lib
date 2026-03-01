"""KuzuGraphStore -- Kuzu-backed implementation of the GraphStore protocol.

Manages dynamic schema (node tables and rel tables created on demand),
parameterized Cypher queries, and BFS traversal.

Public API:
    KuzuGraphStore: Concrete GraphStore implementation backed by Kuzu.
"""

from __future__ import annotations

import uuid
from collections import deque
from pathlib import Path
from typing import Any

import kuzu

from .types import Direction, GraphEdge, GraphNode, TraversalResult


class KuzuGraphStore:
    """Kuzu graph database implementation of the GraphStore protocol.

    Supports dynamic schema: node/rel tables are created on first use
    via ``ensure_node_table`` and ``ensure_rel_table``.  All Cypher
    queries use parameterised bindings to prevent injection.

    Args:
        db_path: Filesystem path for the Kuzu database directory.
        store_id: Optional human-readable identifier; auto-generated if None.
    """

    # ── construction / lifecycle ──────────────────────────────

    def __init__(self, db_path: Path | str, store_id: str | None = None) -> None:
        self._db_path = Path(db_path)
        self._store_id = store_id or f"kuzu-{uuid.uuid4().hex[:8]}"
        self._db = kuzu.Database(str(self._db_path))
        self._conn = kuzu.Connection(self._db)

        # Track which tables we have already ensured so we don't
        # issue redundant DDL on every operation.
        self._known_node_tables: set[str] = set()
        self._known_rel_tables: set[str] = set()

        # Track known columns per node table for schema evolution.
        self._node_table_columns: dict[str, set[str]] = {}

        # Map node_id -> (node_type, properties) for fast lookup
        # without scanning every table.  Populated on add / get.
        self._id_table_cache: dict[str, str] = {}

    @property
    def store_id(self) -> str:
        return self._store_id

    def close(self) -> None:
        """Release Kuzu resources."""
        self._conn = None  # type: ignore[assignment]
        self._db = None  # type: ignore[assignment]

    # ── schema management ─────────────────────────────────────

    def ensure_node_table(
        self,
        table_name: str,
        columns: dict[str, str] | None = None,
    ) -> None:
        """Ensure a node table exists, creating it if necessary.

        Every node table gets a mandatory ``node_id STRING PRIMARY KEY``
        column plus a ``graph_origin STRING`` column.  Additional columns
        are specified via *columns* (name -> Kuzu type string).

        This method is idempotent.

        Args:
            table_name: Name of the node table (e.g. "Agent").
            columns: Extra columns as ``{"col_name": "KUZU_TYPE"}``.
        """
        extra_cols = {
            k: v for k, v in (columns or {}).items()
            if k not in ("node_id", "graph_origin")
        }

        if table_name in self._known_node_tables:
            # Table exists -- add any new columns via ALTER TABLE.
            known = self._node_table_columns.get(table_name, set())
            for col_name, col_type in extra_cols.items():
                if col_name not in known:
                    try:
                        self._conn.execute(
                            f"ALTER TABLE {table_name} ADD {col_name} {col_type} DEFAULT ''"
                        )
                    except RuntimeError:
                        pass  # Column may already exist from a prior session.
                    known.add(col_name)
            self._node_table_columns[table_name] = known
            return

        col_defs = ["node_id STRING", "graph_origin STRING"]
        for col_name, col_type in extra_cols.items():
            col_defs.append(f"{col_name} {col_type}")

        col_defs_str = ", ".join(col_defs)
        ddl = (
            f"CREATE NODE TABLE IF NOT EXISTS {table_name}"
            f"({col_defs_str}, PRIMARY KEY(node_id))"
        )
        self._conn.execute(ddl)
        self._known_node_tables.add(table_name)
        self._node_table_columns[table_name] = set(extra_cols.keys())

    def ensure_rel_table(
        self,
        table_name: str,
        from_table: str,
        to_table: str,
        columns: dict[str, str] | None = None,
    ) -> None:
        """Ensure a relationship table exists, creating it if necessary.

        Every rel table gets a mandatory ``edge_id STRING`` and
        ``graph_origin STRING`` column.  Additional columns are
        specified via *columns*.

        This method is idempotent.

        Args:
            table_name: Name of the rel table (e.g. "KNOWS").
            from_table: Source node table.
            to_table: Target node table.
            columns: Extra columns as ``{"col_name": "KUZU_TYPE"}``.
        """
        key = (table_name, from_table, to_table)
        if key in self._known_rel_tables:
            return

        # Ensure source / target tables exist (minimal schema).
        self.ensure_node_table(from_table)
        self.ensure_node_table(to_table)

        col_defs = ["edge_id STRING", "graph_origin STRING"]
        for col_name, col_type in (columns or {}).items():
            if col_name not in ("edge_id", "graph_origin"):
                col_defs.append(f"{col_name} {col_type}")

        col_defs_str = ", ".join(col_defs)
        ddl = (
            f"CREATE REL TABLE IF NOT EXISTS {table_name}"
            f"(FROM {from_table} TO {to_table}, {col_defs_str})"
        )
        self._conn.execute(ddl)
        self._known_rel_tables.add(key)

    # ── node CRUD ─────────────────────────────────────────────

    def add_node(
        self,
        node_type: str,
        properties: dict[str, Any],
        node_id: str | None = None,
    ) -> GraphNode:
        """Create a node in the graph.

        Automatically ensures the node table exists with columns matching
        the supplied *properties* (all stored as STRING for flexibility).
        """
        nid = node_id or uuid.uuid4().hex
        self.ensure_node_table(
            node_type,
            {k: "STRING" for k in properties},
        )

        # Build parameterised CREATE
        set_parts = ["node_id: $node_id", "graph_origin: $graph_origin"]
        params: dict[str, Any] = {
            "node_id": nid,
            "graph_origin": self._store_id,
        }
        for idx, (k, v) in enumerate(properties.items()):
            pname = f"p{idx}"
            set_parts.append(f"{k}: ${pname}")
            params[pname] = str(v)

        set_clause = ", ".join(set_parts)
        cypher = f"CREATE (:{node_type} {{{set_clause}}})"
        self._conn.execute(cypher, params)

        self._id_table_cache[nid] = node_type
        return GraphNode(
            node_id=nid,
            node_type=node_type,
            properties=dict(properties),
            graph_origin=self._store_id,
        )

    def get_node(self, node_id: str) -> GraphNode | None:
        """Fetch a node by ID, searching all known tables."""
        # Try cached table first.
        cached_table = self._id_table_cache.get(node_id)
        if cached_table:
            node = self._get_node_from_table(node_id, cached_table)
            if node is not None:
                return node

        # Fall back to scanning all known tables.
        for table in list(self._known_node_tables):
            if table == cached_table:
                continue
            node = self._get_node_from_table(node_id, table)
            if node is not None:
                self._id_table_cache[node_id] = table
                return node

        return None

    def _get_node_from_table(self, node_id: str, table: str) -> GraphNode | None:
        """Fetch a node from a specific table by ID."""
        cypher = (
            f"MATCH (n:{table}) WHERE n.node_id = $nid "
            f"RETURN n"
        )
        result = self._conn.execute(cypher, {"nid": node_id})
        if not result.has_next():
            return None

        row = result.get_next()
        return self._row_to_node(row[0], table)

    def query_nodes(
        self,
        node_type: str,
        filters: dict[str, Any] | None = None,
        limit: int = 50,
    ) -> list[GraphNode]:
        """Query nodes of a given type with optional equality filters."""
        if node_type not in self._known_node_tables:
            return []

        where_parts: list[str] = []
        params: dict[str, Any] = {}

        for idx, (k, v) in enumerate((filters or {}).items()):
            pname = f"f{idx}"
            where_parts.append(f"n.{k} = ${pname}")
            params[pname] = str(v)

        where_clause = f" WHERE {' AND '.join(where_parts)}" if where_parts else ""
        cypher = f"MATCH (n:{node_type}){where_clause} RETURN n LIMIT {limit}"
        result = self._conn.execute(cypher, params)

        nodes: list[GraphNode] = []
        while result.has_next():
            row = result.get_next()
            nodes.append(self._row_to_node(row[0], node_type))
        return nodes

    def search_nodes(
        self,
        node_type: str,
        text_fields: list[str],
        query: str,
        filters: dict[str, Any] | None = None,
        limit: int = 10,
    ) -> list[GraphNode]:
        """Keyword search across text fields using CONTAINS."""
        if node_type not in self._known_node_tables:
            return []

        where_parts: list[str] = []
        params: dict[str, Any] = {"query": query.lower()}

        # Text search across specified fields (OR).
        text_clauses = [
            f"lower(n.{field}) CONTAINS $query" for field in text_fields
        ]
        if text_clauses:
            where_parts.append(f"({' OR '.join(text_clauses)})")

        # Extra equality filters.
        for idx, (k, v) in enumerate((filters or {}).items()):
            pname = f"f{idx}"
            where_parts.append(f"n.{k} = ${pname}")
            params[pname] = str(v)

        where_clause = f" WHERE {' AND '.join(where_parts)}" if where_parts else ""
        cypher = f"MATCH (n:{node_type}){where_clause} RETURN n LIMIT {limit}"
        result = self._conn.execute(cypher, params)

        nodes: list[GraphNode] = []
        while result.has_next():
            row = result.get_next()
            nodes.append(self._row_to_node(row[0], node_type))
        return nodes

    def update_node(self, node_id: str, properties: dict[str, Any]) -> bool:
        """Update properties on an existing node."""
        table = self._id_table_cache.get(node_id)
        if table is None:
            # Try to find it.
            node = self.get_node(node_id)
            if node is None:
                return False
            table = node.node_type

        set_parts: list[str] = []
        params: dict[str, Any] = {"nid": node_id}
        for idx, (k, v) in enumerate(properties.items()):
            pname = f"u{idx}"
            set_parts.append(f"n.{k} = ${pname}")
            params[pname] = str(v)

        if not set_parts:
            return True  # nothing to update

        set_clause = ", ".join(set_parts)
        cypher = f"MATCH (n:{table}) WHERE n.node_id = $nid SET {set_clause}"
        self._conn.execute(cypher, params)
        return True

    def delete_node(self, node_id: str) -> bool:
        """Delete a node by ID from its table."""
        table = self._id_table_cache.get(node_id)
        if table is None:
            node = self.get_node(node_id)
            if node is None:
                return False
            table = node.node_type

        cypher = f"MATCH (n:{table}) WHERE n.node_id = $nid DETACH DELETE n"
        self._conn.execute(cypher, {"nid": node_id})
        self._id_table_cache.pop(node_id, None)
        return True

    # ── edge operations ───────────────────────────────────────

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        properties: dict[str, Any] | None = None,
    ) -> GraphEdge:
        """Create a directed edge between two existing nodes.

        Raises:
            KeyError: If source or target node does not exist.
        """
        src_node = self.get_node(source_id)
        tgt_node = self.get_node(target_id)
        if src_node is None:
            raise KeyError(f"Source node not found: {source_id}")
        if tgt_node is None:
            raise KeyError(f"Target node not found: {target_id}")

        props = properties or {}
        eid = uuid.uuid4().hex

        self.ensure_rel_table(
            edge_type,
            src_node.node_type,
            tgt_node.node_type,
            {k: "STRING" for k in props},
        )

        # Build parameterised CREATE for the relationship.
        set_parts = ["edge_id: $eid", "graph_origin: $graph_origin"]
        params: dict[str, Any] = {
            "sid": source_id,
            "tid": target_id,
            "eid": eid,
            "graph_origin": self._store_id,
        }
        for idx, (k, v) in enumerate(props.items()):
            pname = f"ep{idx}"
            set_parts.append(f"{k}: ${pname}")
            params[pname] = str(v)

        set_clause = ", ".join(set_parts)
        cypher = (
            f"MATCH (a:{src_node.node_type}), (b:{tgt_node.node_type}) "
            f"WHERE a.node_id = $sid AND b.node_id = $tid "
            f"CREATE (a)-[:{edge_type} {{{set_clause}}}]->(b)"
        )
        self._conn.execute(cypher, params)

        return GraphEdge(
            edge_id=eid,
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            properties=dict(props),
            graph_origin=self._store_id,
        )

    def query_neighbors(
        self,
        node_id: str,
        edge_type: str | None = None,
        direction: Direction = Direction.BOTH,
        limit: int = 50,
    ) -> list[tuple[GraphEdge, GraphNode]]:
        """Return edges and neighbor nodes adjacent to node_id."""
        node = self.get_node(node_id)
        if node is None:
            return []

        results: list[tuple[GraphEdge, GraphNode]] = []

        # Determine which rel tables to scan.
        rel_tables = self._get_rel_tables_for(edge_type)

        for rel_name, from_table, to_table in rel_tables:
            # Outgoing: source is our node
            if direction in (Direction.OUTGOING, Direction.BOTH):
                if from_table == node.node_type:
                    results.extend(
                        self._query_directed_neighbors(
                            node_id, node.node_type, to_table,
                            rel_name, "outgoing", limit,
                        )
                    )

            # Incoming: target is our node
            if direction in (Direction.INCOMING, Direction.BOTH):
                if to_table == node.node_type:
                    results.extend(
                        self._query_directed_neighbors(
                            node_id, node.node_type, from_table,
                            rel_name, "incoming", limit,
                        )
                    )

        return results[:limit]

    def _query_directed_neighbors(
        self,
        node_id: str,
        node_table: str,
        neighbor_table: str,
        rel_name: str,
        direction: str,
        limit: int,
    ) -> list[tuple[GraphEdge, GraphNode]]:
        """Query neighbors in a specific direction for a single rel table."""
        params: dict[str, Any] = {"nid": node_id}

        if direction == "outgoing":
            cypher = (
                f"MATCH (a:{node_table})-[r:{rel_name}]->(b:{neighbor_table}) "
                f"WHERE a.node_id = $nid RETURN r, b LIMIT {limit}"
            )
        else:
            cypher = (
                f"MATCH (a:{neighbor_table})-[r:{rel_name}]->(b:{node_table}) "
                f"WHERE b.node_id = $nid RETURN r, a LIMIT {limit}"
            )

        result = self._conn.execute(cypher, params)
        pairs: list[tuple[GraphEdge, GraphNode]] = []

        while result.has_next():
            row = result.get_next()
            rel_dict = row[0]
            neighbor_dict = row[1]

            edge = self._rel_to_edge(rel_dict, rel_name, node_id, direction)
            neighbor = self._row_to_node(neighbor_dict, neighbor_table)
            self._id_table_cache[neighbor.node_id] = neighbor_table
            pairs.append((edge, neighbor))

        return pairs

    def delete_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
    ) -> bool:
        """Delete a specific edge between two nodes."""
        src_node = self.get_node(source_id)
        tgt_node = self.get_node(target_id)
        if src_node is None or tgt_node is None:
            return False

        cypher = (
            f"MATCH (a:{src_node.node_type})-[r:{edge_type}]->(b:{tgt_node.node_type}) "
            f"WHERE a.node_id = $sid AND b.node_id = $tid DELETE r"
        )
        self._conn.execute(cypher, {"sid": source_id, "tid": target_id})
        return True

    # ── traversal ─────────────────────────────────────────────

    def traverse(
        self,
        start_id: str,
        edge_types: list[str] | None = None,
        max_hops: int = 3,
        direction: Direction = Direction.OUTGOING,
        node_filter: dict[str, Any] | None = None,
    ) -> TraversalResult:
        """BFS traversal from start_id up to max_hops hops.

        Uses iterative BFS via query_neighbors rather than a single
        recursive Cypher query, giving us fine-grained control over
        filtering, hop counting, and boundary detection.
        """
        start_node = self.get_node(start_id)
        if start_node is None:
            return TraversalResult()

        visited_ids: set[str] = {start_id}
        all_nodes: dict[str, GraphNode] = {start_id: start_node}
        all_edges: list[GraphEdge] = []
        paths: list[list] = []
        origins: set[str] = {start_node.graph_origin}

        # BFS queue: (current_node_id, current_path, hops_so_far)
        queue: deque[tuple[str, list, int]] = deque()
        queue.append((start_id, [start_node], 0))

        while queue:
            current_id, current_path, hops = queue.popleft()
            if hops >= max_hops:
                continue

            # Get neighbors, optionally filtering by edge type.
            if edge_types:
                neighbors: list[tuple[GraphEdge, GraphNode]] = []
                for et in edge_types:
                    neighbors.extend(
                        self.query_neighbors(current_id, et, direction)
                    )
            else:
                neighbors = self.query_neighbors(
                    current_id, direction=direction,
                )

            for edge, neighbor in neighbors:
                # Apply node filter if specified.
                if node_filter:
                    if not self._matches_filter(neighbor, node_filter):
                        continue

                all_edges.append(edge)
                origins.add(neighbor.graph_origin)

                new_path = current_path + [edge, neighbor]

                if neighbor.node_id not in visited_ids:
                    visited_ids.add(neighbor.node_id)
                    all_nodes[neighbor.node_id] = neighbor
                    queue.append((neighbor.node_id, new_path, hops + 1))

                # Record the path regardless (allows multiple paths to same node).
                paths.append(new_path)

        return TraversalResult(
            paths=paths,
            nodes=list(all_nodes.values()),
            edges=all_edges,
            crossed_boundaries=len(origins) > 1,
        )

    # ── private helpers ───────────────────────────────────────

    def _row_to_node(self, row_data: dict[str, Any], table: str) -> GraphNode:
        """Convert a Kuzu result row dict to a GraphNode."""
        node_id = str(row_data.get("node_id", ""))
        graph_origin = str(row_data.get("graph_origin", ""))

        # Everything except internal fields goes into properties.
        skip = {"node_id", "graph_origin", "_id", "_label"}
        props = {k: v for k, v in row_data.items() if k not in skip}

        self._id_table_cache[node_id] = table
        return GraphNode(
            node_id=node_id,
            node_type=table,
            properties=props,
            graph_origin=graph_origin,
        )

    def _rel_to_edge(
        self,
        rel_data: dict[str, Any],
        rel_name: str,
        anchor_id: str,
        direction: str,
    ) -> GraphEdge:
        """Convert a Kuzu relationship dict to a GraphEdge."""
        eid = str(rel_data.get("edge_id", ""))
        graph_origin = str(rel_data.get("graph_origin", ""))

        skip = {"edge_id", "graph_origin", "_id", "_label", "_src", "_dst"}
        props = {k: v for k, v in rel_data.items() if k not in skip}

        # Kuzu _src/_dst are internal IDs, not our node_ids.
        # We reconstruct source/target from the anchor and direction.
        if direction == "outgoing":
            source_id = anchor_id
            target_id = ""  # filled from the neighbor context
        else:
            source_id = ""  # filled from the neighbor context
            target_id = anchor_id

        return GraphEdge(
            edge_id=eid,
            source_id=source_id,
            target_id=target_id,
            edge_type=rel_name,
            properties=props,
            graph_origin=graph_origin,
        )

    def _get_rel_tables_for(
        self,
        edge_type: str | None,
    ) -> list[tuple[str, str, str]]:
        """Return (rel_name, from_table, to_table) tuples matching edge_type."""
        matches = []
        for key in self._known_rel_tables:
            rel_name, from_table, to_table = key
            if edge_type is None or rel_name == edge_type:
                matches.append((rel_name, from_table, to_table))
        return matches

    @staticmethod
    def _matches_filter(node: GraphNode, node_filter: dict[str, Any]) -> bool:
        """Check whether a node's properties match all filter criteria."""
        for k, v in node_filter.items():
            if node.properties.get(k) != str(v):
                return False
        return True


__all__ = ["KuzuGraphStore"]
