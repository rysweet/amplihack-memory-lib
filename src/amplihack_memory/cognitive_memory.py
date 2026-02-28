"""Cognitive memory system with six memory types.

Provides a single CognitiveMemory class that manages all six cognitive memory
types (sensory, working, episodic, semantic, procedural, prospective) backed
by a Kuzu graph database.

Each agent gets full isolation via an ``agent_id`` column stored in every
node table.
"""

import json
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import kuzu

from .memory_types import (
    ConsolidatedEpisode,
    EpisodicMemory,
    MemoryCategory,
    ProceduralMemory,
    ProspectiveMemory,
    SemanticFact,
    SensoryItem,
    WorkingMemorySlot,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WORKING_MEMORY_CAPACITY = 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_id(prefix: str = "mem") -> str:
    """Generate a short unique id with a human-readable prefix."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _ts_now() -> int:
    """Current Unix timestamp as integer."""
    return int(time.time())


def _safe_execute(conn: kuzu.Connection, query: str, params: dict | None = None):
    """Execute a Kuzu query, returning the QueryResult or None on error."""
    if params is None:
        params = {}
    return conn.execute(query, params)


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------

_NODE_TABLES = [
    """
    CREATE NODE TABLE IF NOT EXISTS SensoryMemory(
        node_id STRING,
        agent_id STRING,
        modality STRING,
        raw_data STRING,
        observation_order INT64,
        expires_at INT64,
        created_at INT64,
        PRIMARY KEY(node_id)
    )
    """,
    """
    CREATE NODE TABLE IF NOT EXISTS WorkingMemory(
        node_id STRING,
        agent_id STRING,
        slot_type STRING,
        content STRING,
        relevance DOUBLE,
        task_id STRING,
        created_at INT64,
        PRIMARY KEY(node_id)
    )
    """,
    """
    CREATE NODE TABLE IF NOT EXISTS EpisodicMemory(
        node_id STRING,
        agent_id STRING,
        content STRING,
        source_label STRING,
        temporal_index INT64,
        compressed BOOLEAN,
        metadata STRING,
        created_at INT64,
        PRIMARY KEY(node_id)
    )
    """,
    """
    CREATE NODE TABLE IF NOT EXISTS SemanticMemory(
        node_id STRING,
        agent_id STRING,
        concept STRING,
        content STRING,
        confidence DOUBLE,
        source_id STRING,
        tags STRING,
        metadata STRING,
        created_at INT64,
        entity_name STRING DEFAULT '',
        PRIMARY KEY(node_id)
    )
    """,
    """
    CREATE NODE TABLE IF NOT EXISTS ProceduralMemory(
        node_id STRING,
        agent_id STRING,
        name STRING,
        steps STRING,
        prerequisites STRING,
        usage_count INT64,
        created_at INT64,
        PRIMARY KEY(node_id)
    )
    """,
    """
    CREATE NODE TABLE IF NOT EXISTS ProspectiveMemory(
        node_id STRING,
        agent_id STRING,
        desc_text STRING,
        trigger_condition STRING,
        action_on_trigger STRING,
        status STRING,
        priority INT64,
        created_at INT64,
        PRIMARY KEY(node_id)
    )
    """,
    """
    CREATE NODE TABLE IF NOT EXISTS ConsolidatedEpisode(
        node_id STRING,
        agent_id STRING,
        summary STRING,
        original_count INT64,
        created_at INT64,
        PRIMARY KEY(node_id)
    )
    """,
]

_REL_TABLES = [
    """
    CREATE REL TABLE IF NOT EXISTS SIMILAR_TO(
        FROM SemanticMemory TO SemanticMemory,
        similarity_score DOUBLE
    )
    """,
    """
    CREATE REL TABLE IF NOT EXISTS DERIVES_FROM(
        FROM SemanticMemory TO EpisodicMemory,
        derived_at INT64
    )
    """,
    """
    CREATE REL TABLE IF NOT EXISTS PROCEDURE_DERIVES_FROM(
        FROM ProceduralMemory TO EpisodicMemory,
        derived_at INT64
    )
    """,
    """
    CREATE REL TABLE IF NOT EXISTS CONSOLIDATES(
        FROM ConsolidatedEpisode TO EpisodicMemory,
        consolidated_at INT64
    )
    """,
    """
    CREATE REL TABLE IF NOT EXISTS ATTENDED_TO(
        FROM SensoryMemory TO EpisodicMemory,
        attended_at INT64
    )
    """,
    """
    CREATE REL TABLE IF NOT EXISTS SUPERSEDES(
        FROM SemanticMemory TO SemanticMemory,
        reason STRING,
        temporal_delta STRING
    )
    """,
]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class CognitiveMemory:
    """Six-type cognitive memory backed by Kuzu.

    Args:
        agent_name: Identifier for the agent (used for row isolation).
        db_path: Filesystem path for the Kuzu database directory.
    """

    WORKING_MEMORY_CAPACITY = WORKING_MEMORY_CAPACITY

    def __init__(self, agent_name: str, db_path: str | Path) -> None:
        if not agent_name or not agent_name.strip():
            raise ValueError("agent_name cannot be empty")

        self.agent_name = agent_name.strip()
        self.db_path = Path(db_path)
        # Ensure the *parent* directory exists; Kuzu creates the db dir itself.
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._db = kuzu.Database(str(self.db_path))
        self._conn = kuzu.Connection(self._db)
        self._initialize_schema()

        # Monotonic counters (loaded from DB at startup)
        self._sensory_order = self._load_max_order("SensoryMemory", "observation_order")
        self._temporal_index = self._load_max_order("EpisodicMemory", "temporal_index")

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _initialize_schema(self) -> None:
        for ddl in _NODE_TABLES + _REL_TABLES:
            try:
                self._conn.execute(ddl)
            except Exception as exc:
                if "already exists" not in str(exc).lower():
                    raise

    def _load_max_order(self, table: str, column: str) -> int:
        result = self._conn.execute(
            f"MATCH (n:{table}) WHERE n.agent_id = $aid "
            f"RETURN max(n.{column})",
            {"aid": self.agent_name},
        )
        if result.has_next():
            val = result.get_next()[0]
            if val is not None:
                return int(val)
        return 0

    # ==================================================================
    # SENSORY MEMORY
    # ==================================================================

    def record_sensory(
        self,
        modality: str,
        raw_data: str,
        ttl_seconds: int = 300,
    ) -> str:
        """Record a short-lived sensory observation.

        Args:
            modality: Channel name (e.g. "text", "code", "error").
            raw_data: The raw observation payload.
            ttl_seconds: Time-to-live in seconds (default 300 = 5 min).

        Returns:
            The node_id of the newly created sensory item.
        """
        node_id = _new_id("sen")
        now = _ts_now()
        self._sensory_order += 1
        self._conn.execute(
            """
            CREATE (:SensoryMemory {
                node_id: $nid,
                agent_id: $aid,
                modality: $mod,
                raw_data: $data,
                observation_order: $ord,
                expires_at: $exp,
                created_at: $ts
            })
            """,
            {
                "nid": node_id,
                "aid": self.agent_name,
                "mod": modality,
                "data": raw_data,
                "ord": self._sensory_order,
                "exp": now + ttl_seconds,
                "ts": now,
            },
        )
        return node_id

    def get_recent_sensory(self, limit: int = 10) -> list[SensoryItem]:
        """Return the most recent sensory items (not yet expired).

        Args:
            limit: Maximum items to return.

        Returns:
            List of SensoryItem ordered by observation_order descending.
        """
        now = _ts_now()
        result = self._conn.execute(
            """
            MATCH (s:SensoryMemory)
            WHERE s.agent_id = $aid AND s.expires_at > $now
            RETURN s.node_id, s.modality, s.raw_data,
                   s.observation_order, s.expires_at, s.created_at
            ORDER BY s.observation_order DESC
            LIMIT $lim
            """,
            {"aid": self.agent_name, "now": now, "lim": limit},
        )
        items: list[SensoryItem] = []
        while result.has_next():
            row = result.get_next()
            items.append(SensoryItem(
                node_id=row[0],
                modality=row[1],
                raw_data=row[2],
                observation_order=int(row[3]),
                expires_at=float(row[4]),
                created_at=datetime.fromtimestamp(row[5]),
            ))
        return items

    def attend_to_sensory(self, sensory_id: str, reason: str) -> str | None:
        """Promote a sensory item to an episodic memory and link them.

        Args:
            sensory_id: node_id of the sensory item to promote.
            reason: Why this observation is noteworthy.

        Returns:
            The node_id of the new episodic memory, or None if the
            sensory item was not found / already expired.
        """
        now = _ts_now()
        # Fetch the sensory item
        result = self._conn.execute(
            """
            MATCH (s:SensoryMemory)
            WHERE s.node_id = $sid AND s.agent_id = $aid AND s.expires_at > $now
            RETURN s.modality, s.raw_data
            """,
            {"sid": sensory_id, "aid": self.agent_name, "now": now},
        )
        if not result.has_next():
            return None

        modality, raw_data = result.get_next()
        content = f"[{modality}] {raw_data} -- attended: {reason}"

        ep_id = self.store_episode(content=content, source_label="sensory-attention")

        # Create ATTENDED_TO edge
        try:
            self._conn.execute(
                """
                MATCH (s:SensoryMemory), (e:EpisodicMemory)
                WHERE s.node_id = $sid AND e.node_id = $eid
                CREATE (s)-[:ATTENDED_TO {attended_at: $ts}]->(e)
                """,
                {"sid": sensory_id, "eid": ep_id, "ts": now},
            )
        except Exception:
            pass  # edge creation is best-effort

        return ep_id

    def prune_expired_sensory(self) -> int:
        """Delete sensory items past their expiry time.

        Returns:
            Number of items pruned.
        """
        now = _ts_now()
        # Count first
        result = self._conn.execute(
            """
            MATCH (s:SensoryMemory)
            WHERE s.agent_id = $aid AND s.expires_at <= $now
            RETURN count(s)
            """,
            {"aid": self.agent_name, "now": now},
        )
        count = 0
        if result.has_next():
            count = int(result.get_next()[0])

        if count > 0:
            self._conn.execute(
                """
                MATCH (s:SensoryMemory)
                WHERE s.agent_id = $aid AND s.expires_at <= $now
                DELETE s
                """,
                {"aid": self.agent_name, "now": now},
            )
        return count

    # ==================================================================
    # WORKING MEMORY
    # ==================================================================

    def push_working(
        self,
        slot_type: str,
        content: str,
        task_id: str,
        relevance: float = 1.0,
    ) -> str:
        """Push a slot into working memory for a given task.

        If the task already has WORKING_MEMORY_CAPACITY slots, the
        least-relevant slot is evicted.

        Args:
            slot_type: Categorisation (e.g. "goal", "constraint").
            content: Payload.
            task_id: Task identifier.
            relevance: Priority weight (default 1.0).

        Returns:
            node_id of the new working-memory slot.
        """
        # Evict if at capacity
        existing = self.get_working(task_id)
        if len(existing) >= self.WORKING_MEMORY_CAPACITY:
            # Remove the lowest-relevance slot
            lowest = min(existing, key=lambda s: s.relevance)
            self._conn.execute(
                "MATCH (w:WorkingMemory) WHERE w.node_id = $nid DELETE w",
                {"nid": lowest.node_id},
            )

        node_id = _new_id("wrk")
        now = _ts_now()
        self._conn.execute(
            """
            CREATE (:WorkingMemory {
                node_id: $nid,
                agent_id: $aid,
                slot_type: $st,
                content: $cnt,
                relevance: $rel,
                task_id: $tid,
                created_at: $ts
            })
            """,
            {
                "nid": node_id,
                "aid": self.agent_name,
                "st": slot_type,
                "cnt": content,
                "rel": relevance,
                "tid": task_id,
                "ts": now,
            },
        )
        return node_id

    def get_working(self, task_id: str) -> list[WorkingMemorySlot]:
        """Retrieve working memory slots for a task.

        Args:
            task_id: Task identifier.

        Returns:
            List of WorkingMemorySlot ordered by relevance descending.
        """
        result = self._conn.execute(
            """
            MATCH (w:WorkingMemory)
            WHERE w.agent_id = $aid AND w.task_id = $tid
            RETURN w.node_id, w.slot_type, w.content,
                   w.relevance, w.task_id, w.created_at
            ORDER BY w.relevance DESC
            """,
            {"aid": self.agent_name, "tid": task_id},
        )
        slots: list[WorkingMemorySlot] = []
        while result.has_next():
            row = result.get_next()
            slots.append(WorkingMemorySlot(
                node_id=row[0],
                slot_type=row[1],
                content=row[2],
                relevance=float(row[3]),
                task_id=row[4],
                created_at=datetime.fromtimestamp(row[5]),
            ))
        return slots

    def clear_working(self, task_id: str) -> int:
        """Clear all working memory slots for a task.

        Args:
            task_id: Task identifier.

        Returns:
            Number of slots cleared.
        """
        result = self._conn.execute(
            """
            MATCH (w:WorkingMemory)
            WHERE w.agent_id = $aid AND w.task_id = $tid
            RETURN count(w)
            """,
            {"aid": self.agent_name, "tid": task_id},
        )
        count = 0
        if result.has_next():
            count = int(result.get_next()[0])

        if count > 0:
            self._conn.execute(
                """
                MATCH (w:WorkingMemory)
                WHERE w.agent_id = $aid AND w.task_id = $tid
                DELETE w
                """,
                {"aid": self.agent_name, "tid": task_id},
            )
        return count

    # ==================================================================
    # EPISODIC MEMORY
    # ==================================================================

    def store_episode(
        self,
        content: str,
        source_label: str,
        temporal_index: int | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Store an episodic memory.

        Args:
            content: Episode description.
            source_label: Origin label (e.g. "user-session").
            temporal_index: Explicit ordering index. If None, auto-increments.
            metadata: Optional structured data.

        Returns:
            node_id of the stored episode.
        """
        node_id = _new_id("epi")
        now = _ts_now()
        if temporal_index is None:
            self._temporal_index += 1
            temporal_index = self._temporal_index
        else:
            # Keep our counter up to date
            if temporal_index > self._temporal_index:
                self._temporal_index = temporal_index

        meta_json = json.dumps(metadata) if metadata else "{}"

        self._conn.execute(
            """
            CREATE (:EpisodicMemory {
                node_id: $nid,
                agent_id: $aid,
                content: $cnt,
                source_label: $src,
                temporal_index: $tidx,
                compressed: false,
                metadata: $meta,
                created_at: $ts
            })
            """,
            {
                "nid": node_id,
                "aid": self.agent_name,
                "cnt": content,
                "src": source_label,
                "tidx": temporal_index,
                "meta": meta_json,
                "ts": now,
            },
        )
        return node_id

    def consolidate_episodes(
        self,
        batch_size: int = 10,
        summarizer: Callable[[list[str]], str] | None = None,
    ) -> str | None:
        """Consolidate the oldest un-compressed episodes into a summary.

        Args:
            batch_size: Number of episodes to consolidate per call.
            summarizer: Optional function ``(list[str]) -> str`` that
                produces a summary.  If None a simple concatenation is used.

        Returns:
            node_id of the ConsolidatedEpisode, or None if fewer than
            ``batch_size`` episodes are available.
        """
        result = self._conn.execute(
            """
            MATCH (e:EpisodicMemory)
            WHERE e.agent_id = $aid AND e.compressed = false
            RETURN e.node_id, e.content
            ORDER BY e.temporal_index ASC
            LIMIT $bs
            """,
            {"aid": self.agent_name, "bs": batch_size},
        )
        episodes: list[tuple[str, str]] = []
        while result.has_next():
            row = result.get_next()
            episodes.append((row[0], row[1]))

        if len(episodes) < batch_size:
            return None

        contents = [c for _, c in episodes]
        if summarizer is not None:
            summary = summarizer(contents)
        else:
            summary = " | ".join(contents)

        # Create consolidated node
        cons_id = _new_id("con")
        now = _ts_now()
        self._conn.execute(
            """
            CREATE (:ConsolidatedEpisode {
                node_id: $nid,
                agent_id: $aid,
                summary: $sum,
                original_count: $cnt,
                created_at: $ts
            })
            """,
            {
                "nid": cons_id,
                "aid": self.agent_name,
                "sum": summary,
                "cnt": len(episodes),
                "ts": now,
            },
        )

        # Mark originals as compressed and create edges
        for ep_id, _ in episodes:
            self._conn.execute(
                """
                MATCH (e:EpisodicMemory)
                WHERE e.node_id = $eid
                SET e.compressed = true
                """,
                {"eid": ep_id},
            )
            try:
                self._conn.execute(
                    """
                    MATCH (c:ConsolidatedEpisode), (e:EpisodicMemory)
                    WHERE c.node_id = $cid AND e.node_id = $eid
                    CREATE (c)-[:CONSOLIDATES {consolidated_at: $ts}]->(e)
                    """,
                    {"cid": cons_id, "eid": ep_id, "ts": now},
                )
            except Exception:
                pass  # edge creation is best-effort

        return cons_id

    def get_episodes(
        self,
        limit: int = 20,
        include_compressed: bool = False,
    ) -> list[EpisodicMemory]:
        """Retrieve episodic memories.

        Args:
            limit: Maximum number of episodes.
            include_compressed: If False (default), exclude compressed episodes.

        Returns:
            List of EpisodicMemory sorted by temporal_index descending.
        """
        compressed_clause = "" if include_compressed else " AND e.compressed = false"
        result = self._conn.execute(
            f"""
            MATCH (e:EpisodicMemory)
            WHERE e.agent_id = $aid{compressed_clause}
            RETURN e.node_id, e.content, e.source_label,
                   e.temporal_index, e.compressed, e.created_at, e.metadata
            ORDER BY e.temporal_index DESC
            LIMIT $lim
            """,
            {"aid": self.agent_name, "lim": limit},
        )
        episodes: list[EpisodicMemory] = []
        while result.has_next():
            row = result.get_next()
            meta = {}
            if row[6]:
                try:
                    meta = json.loads(row[6])
                except (json.JSONDecodeError, TypeError):
                    pass
            episodes.append(EpisodicMemory(
                node_id=row[0],
                content=row[1],
                source_label=row[2],
                temporal_index=int(row[3]),
                compressed=bool(row[4]),
                created_at=datetime.fromtimestamp(row[5]),
                metadata=meta,
            ))
        return episodes

    # ==================================================================
    # SEMANTIC MEMORY
    # ==================================================================

    def store_fact(
        self,
        concept: str,
        content: str,
        confidence: float = 1.0,
        source_id: str = "",
        tags: list[str] | None = None,
        temporal_metadata: dict | None = None,
    ) -> str:
        """Store a semantic fact.

        Auto-extracts entity name for entity-centric indexing.
        Detects and creates SUPERSEDES edges when newer temporal data
        updates an existing fact about the same entity.

        Args:
            concept: The concept or topic.
            content: Factual content.
            confidence: Confidence score (0.0 - 1.0).
            source_id: Origin reference.
            tags: Categorisation tags.
            temporal_metadata: Optional structured metadata with keys like
                source_date, temporal_order, temporal_index.

        Returns:
            node_id of the stored fact.
        """
        node_id = _new_id("sem")
        now = _ts_now()
        tags_json = json.dumps(tags) if tags else "[]"
        meta_json = json.dumps(temporal_metadata) if temporal_metadata else "{}"

        # Extract entity name for entity-centric indexing
        entity_name = self._extract_entity_name(content, concept)

        self._conn.execute(
            """
            CREATE (:SemanticMemory {
                node_id: $nid,
                agent_id: $aid,
                concept: $con,
                content: $cnt,
                confidence: $conf,
                source_id: $src,
                tags: $tags,
                metadata: $meta,
                created_at: $ts,
                entity_name: $ename
            })
            """,
            {
                "nid": node_id,
                "aid": self.agent_name,
                "con": concept,
                "cnt": content,
                "conf": confidence,
                "src": source_id,
                "tags": tags_json,
                "meta": meta_json,
                "ts": now,
                "ename": entity_name,
            },
        )

        # Detect and create SUPERSEDES edges for temporal updates
        if temporal_metadata and temporal_metadata.get("temporal_index", 0) > 0:
            self._detect_supersedes(node_id, content, concept, temporal_metadata)

        return node_id

    def search_facts(
        self,
        query: str,
        limit: int = 10,
        min_confidence: float = 0.0,
    ) -> list[SemanticFact]:
        """Search semantic facts by keyword matching on concept and content.

        Args:
            query: Search string (keywords matched via CONTAINS).
            limit: Maximum results.
            min_confidence: Minimum confidence threshold.

        Returns:
            List of matching SemanticFact ordered by confidence descending.
        """
        # Tokenise query and build OR conditions
        keywords = [w.strip() for w in query.split() if w.strip()]
        if not keywords:
            return self.get_all_facts(limit=limit)

        conditions = []
        params: dict = {
            "aid": self.agent_name,
            "minc": min_confidence,
            "lim": limit,
        }
        for i, kw in enumerate(keywords[:6]):
            pname = f"kw{i}"
            conditions.append(
                f"(lower(s.concept) CONTAINS lower(${pname}) "
                f"OR lower(s.content) CONTAINS lower(${pname}))"
            )
            params[pname] = kw

        where_kw = " OR ".join(conditions)
        result = self._conn.execute(
            f"""
            MATCH (s:SemanticMemory)
            WHERE s.agent_id = $aid
              AND s.confidence >= $minc
              AND ({where_kw})
            RETURN s.node_id, s.concept, s.content, s.confidence,
                   s.source_id, s.tags, s.metadata, s.created_at,
                   s.entity_name
            ORDER BY s.confidence DESC
            LIMIT $lim
            """,
            params,
        )
        return self._rows_to_facts(result)

    def get_all_facts(self, limit: int = 50) -> list[SemanticFact]:
        """Return all semantic facts for this agent.

        Args:
            limit: Maximum number of facts.

        Returns:
            List of SemanticFact ordered by confidence descending.
        """
        result = self._conn.execute(
            """
            MATCH (s:SemanticMemory)
            WHERE s.agent_id = $aid
            RETURN s.node_id, s.concept, s.content, s.confidence,
                   s.source_id, s.tags, s.metadata, s.created_at,
                   s.entity_name
            ORDER BY s.confidence DESC
            LIMIT $lim
            """,
            {"aid": self.agent_name, "lim": limit},
        )
        return self._rows_to_facts(result)

    @staticmethod
    def _rows_to_facts(result) -> list[SemanticFact]:
        facts: list[SemanticFact] = []
        while result.has_next():
            row = result.get_next()
            tags: list[str] = []
            if row[5]:
                try:
                    tags = json.loads(row[5])
                except (json.JSONDecodeError, TypeError):
                    pass
            meta: dict = {}
            if row[6]:
                try:
                    meta = json.loads(row[6])
                except (json.JSONDecodeError, TypeError):
                    pass
            entity_name = row[8] if len(row) > 8 else ""
            facts.append(SemanticFact(
                node_id=row[0],
                concept=row[1],
                content=row[2],
                confidence=float(row[3]),
                source_id=row[4] or "",
                tags=tags,
                metadata=meta,
                created_at=datetime.fromtimestamp(row[7]),
                entity_name=entity_name or "",
            ))
        return facts

    # ------------------------------------------------------------------
    # Entity-centric retrieval and concept search
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_entity_name(content: str, concept: str) -> str:
        """Extract the primary entity name from content or concept.

        Uses simple heuristics to find proper nouns (capitalized multi-word
        names) in the concept field first, then the content. Handles
        apostrophe names (O'Brien), hyphenated names (Al-Hassan), and
        multi-word proper nouns (Sarah Chen).

        Args:
            content: Fact content text.
            concept: Concept/topic label.

        Returns:
            Lowercased entity name, or empty string if none found.
        """
        for text in [concept, content]:
            if not text:
                continue
            # Multi-word proper nouns, including apostrophe/hyphenated names
            matches = re.findall(
                r"\b("
                r"[A-Z][a-z]*(?:['\u2019\-][A-Z]?[a-z]+)+(?:\s+(?:[A-Z][a-z]+(?:['\u2019\-][A-Z]?[a-z]+)?))*"
                r"|"
                r"[A-Z][a-z]+(?:\s+(?:[A-Z][a-z]+(?:['\u2019\-][A-Z]?[a-z]+)?))+)"
                r"\b",
                text,
            )
            if matches:
                best = max(matches, key=len)
                return best.lower()

            # Single capitalized word not at start of sentence
            words = text.split()
            for i, word in enumerate(words):
                if i > 0 and word[0:1].isupper() and len(word) > 2:
                    cleaned = word.strip(".,;:!?()[]{}\"'")
                    if cleaned and cleaned[0].isupper():
                        return cleaned.lower()

        return ""

    def retrieve_by_entity(
        self,
        entity_name: str,
        limit: int = 50,
    ) -> list[SemanticFact]:
        """Retrieve ALL facts associated with a specific entity.

        Uses the entity_name index for fast lookup. Falls back to
        content/concept text search if entity_name field is empty
        (backward compatibility with facts stored before entity extraction).

        Args:
            entity_name: Entity name to search for (case-insensitive).
            limit: Maximum nodes to return.

        Returns:
            List of SemanticFact matching the entity.
        """
        if not entity_name or not entity_name.strip():
            return []

        entity_lower = entity_name.strip().lower()

        # Primary: use entity_name index
        result = self._conn.execute(
            """
            MATCH (s:SemanticMemory)
            WHERE s.agent_id = $aid
              AND LOWER(s.entity_name) CONTAINS $entity
            RETURN s.node_id, s.concept, s.content, s.confidence,
                   s.source_id, s.tags, s.metadata, s.created_at,
                   s.entity_name
            ORDER BY s.created_at DESC
            LIMIT $lim
            """,
            {"aid": self.agent_name, "entity": entity_lower, "lim": limit},
        )
        facts = self._rows_to_facts(result)

        # Fallback: text search on content/concept
        if not facts:
            result = self._conn.execute(
                """
                MATCH (s:SemanticMemory)
                WHERE s.agent_id = $aid
                  AND (LOWER(s.content) CONTAINS $entity
                       OR LOWER(s.concept) CONTAINS $entity)
                RETURN s.node_id, s.concept, s.content, s.confidence,
                       s.source_id, s.tags, s.metadata, s.created_at,
                       s.entity_name
                ORDER BY s.created_at DESC
                LIMIT $lim
                """,
                {"aid": self.agent_name, "entity": entity_lower, "lim": limit},
            )
            facts = self._rows_to_facts(result)

        return facts

    def search_by_concept(
        self,
        keywords: list[str],
        limit: int = 30,
    ) -> list[SemanticFact]:
        """Search for facts by concept/content keyword matching.

        Runs a Cypher query matching keywords against both the concept and
        content fields (case-insensitive). Useful for concept-based retrieval
        when entity names are not available in the question.

        Args:
            keywords: List of keyword strings to search for.
            limit: Maximum nodes to return per keyword.

        Returns:
            List of SemanticFact matching any of the keywords, deduplicated.
        """
        if not keywords:
            return []

        facts: list[SemanticFact] = []
        seen: set[str] = set()

        for kw in keywords:
            kw_lower = kw.strip().lower()
            if len(kw_lower) <= 2:
                continue
            result = self._conn.execute(
                """
                MATCH (s:SemanticMemory)
                WHERE s.agent_id = $aid
                  AND (LOWER(s.concept) CONTAINS $kw
                       OR LOWER(s.content) CONTAINS $kw)
                RETURN s.node_id, s.concept, s.content, s.confidence,
                       s.source_id, s.tags, s.metadata, s.created_at,
                       s.entity_name
                ORDER BY s.created_at DESC
                LIMIT $lim
                """,
                {"aid": self.agent_name, "kw": kw_lower, "lim": limit},
            )
            for fact in self._rows_to_facts(result):
                if fact.node_id not in seen:
                    seen.add(fact.node_id)
                    facts.append(fact)

        return facts

    def execute_aggregation(
        self,
        query_type: str,
        entity_filter: str = "",
    ) -> dict[str, Any]:
        """Execute Cypher aggregation queries for meta-memory questions.

        Supports counting, listing, and enumerating entities stored in memory.
        Bypasses text search entirely and uses graph aggregation.

        Args:
            query_type: Type of aggregation:
                - "count_entities": Count distinct entity names
                - "count_concepts": Count distinct concept values
                - "count_by_concept": Count facts grouped by concept
                - "list_entities": List all distinct entity names
                - "list_concepts": List all distinct concept values
                - "count_total": Total number of facts
            entity_filter: Optional filter string for narrowing results.

        Returns:
            Dict with aggregation results.
        """
        try:
            if query_type == "count_total":
                result = self._conn.execute(
                    "MATCH (s:SemanticMemory) WHERE s.agent_id = $aid RETURN COUNT(s)",
                    {"aid": self.agent_name},
                )
                if result.has_next():
                    return {"count": result.get_next()[0], "query_type": query_type}

            elif query_type == "count_entities":
                result = self._conn.execute(
                    """
                    MATCH (s:SemanticMemory)
                    WHERE s.agent_id = $aid AND s.entity_name <> ''
                    RETURN COUNT(DISTINCT s.entity_name)
                    """,
                    {"aid": self.agent_name},
                )
                if result.has_next():
                    return {"count": result.get_next()[0], "query_type": query_type}

            elif query_type == "list_entities":
                result = self._conn.execute(
                    """
                    MATCH (s:SemanticMemory)
                    WHERE s.agent_id = $aid AND s.entity_name <> ''
                    RETURN DISTINCT s.entity_name
                    ORDER BY s.entity_name
                    """,
                    {"aid": self.agent_name},
                )
                items: list[str] = []
                while result.has_next():
                    items.append(result.get_next()[0])
                return {"count": len(items), "items": items, "query_type": query_type}

            elif query_type == "count_concepts":
                result = self._conn.execute(
                    """
                    MATCH (s:SemanticMemory)
                    WHERE s.agent_id = $aid AND s.concept <> ''
                          AND s.concept <> 'SUMMARY'
                    RETURN COUNT(DISTINCT s.concept)
                    """,
                    {"aid": self.agent_name},
                )
                if result.has_next():
                    return {"count": result.get_next()[0], "query_type": query_type}

            elif query_type == "list_concepts":
                filter_clause = ""
                params: dict[str, Any] = {"aid": self.agent_name}
                if entity_filter:
                    filter_clause = " AND LOWER(s.concept) CONTAINS $filter"
                    params["filter"] = entity_filter.lower()

                result = self._conn.execute(
                    f"""
                    MATCH (s:SemanticMemory)
                    WHERE s.agent_id = $aid AND s.concept <> ''
                          AND s.concept <> 'SUMMARY'
                    {filter_clause}
                    RETURN DISTINCT s.concept
                    ORDER BY s.concept
                    """,
                    params,
                )
                items = []
                while result.has_next():
                    items.append(result.get_next()[0])
                return {"count": len(items), "items": items, "query_type": query_type}

            elif query_type == "count_by_concept":
                filter_clause = ""
                params = {"aid": self.agent_name}
                if entity_filter:
                    filter_clause = " AND LOWER(s.concept) CONTAINS $filter"
                    params["filter"] = entity_filter.lower()

                result = self._conn.execute(
                    f"""
                    MATCH (s:SemanticMemory)
                    WHERE s.agent_id = $aid AND s.concept <> ''
                          AND s.concept <> 'SUMMARY'
                    {filter_clause}
                    RETURN s.concept, COUNT(s) AS cnt
                    ORDER BY cnt DESC
                    """,
                    params,
                )
                concept_counts: dict[str, int] = {}
                while result.has_next():
                    row = result.get_next()
                    concept_counts[row[0]] = row[1]
                return {
                    "count": len(concept_counts),
                    "items": concept_counts,
                    "total_facts": sum(concept_counts.values()),
                    "query_type": query_type,
                }

        except Exception:
            pass

        return {"count": 0, "query_type": query_type, "error": "Query failed"}

    # ------------------------------------------------------------------
    # Temporal supersede detection
    # ------------------------------------------------------------------

    def _detect_supersedes(
        self,
        new_node_id: str,
        content: str,
        concept: str,
        temporal_metadata: dict,
    ) -> None:
        """Detect if a new fact supersedes an existing fact about the same entity.

        At STORAGE time, checks for older facts with the same concept that have
        a lower temporal_index. If found with conflicting numbers, creates a
        SUPERSEDES edge (new -> old).

        Args:
            new_node_id: ID of the newly stored fact.
            content: Content of the new fact.
            concept: Concept label of the new fact.
            temporal_metadata: Must contain temporal_index > 0.
        """
        new_temporal_idx = temporal_metadata.get("temporal_index", 0)
        if new_temporal_idx <= 0:
            return

        try:
            concept_key = concept.split()[0] if concept else ""
            if not concept_key:
                return

            result = self._conn.execute(
                """
                MATCH (s:SemanticMemory)
                WHERE s.agent_id = $aid
                  AND s.node_id <> $new_id
                  AND (LOWER(s.concept) CONTAINS LOWER($ckey)
                       OR LOWER($ckey) CONTAINS LOWER(s.concept))
                RETURN s.node_id, s.content, s.concept, s.metadata
                LIMIT 20
                """,
                {
                    "aid": self.agent_name,
                    "new_id": new_node_id,
                    "ckey": concept_key,
                },
            )

            while result.has_next():
                row = result.get_next()
                old_id = row[0]
                old_content = row[1]
                old_meta_str = row[3]

                old_meta = json.loads(old_meta_str) if old_meta_str else {}
                old_temporal_idx = old_meta.get("temporal_index", 0)

                if old_temporal_idx <= 0 or old_temporal_idx >= new_temporal_idx:
                    continue

                contradiction = self._detect_contradiction(
                    content, old_content, concept, row[2]
                )
                if contradiction.get("contradiction"):
                    temporal_delta = f"index {old_temporal_idx} -> {new_temporal_idx}"
                    self._conn.execute(
                        """
                        MATCH (new_s:SemanticMemory {node_id: $new_id})
                        MATCH (old_s:SemanticMemory {node_id: $old_id})
                        CREATE (new_s)-[:SUPERSEDES {
                            reason: $reason,
                            temporal_delta: $delta
                        }]->(old_s)
                        """,
                        {
                            "new_id": new_node_id,
                            "old_id": old_id,
                            "reason": f"Updated values: {contradiction.get('conflicting_values', '')}",
                            "delta": temporal_delta,
                        },
                    )
        except Exception:
            pass  # supersede detection is best-effort

    @staticmethod
    def _detect_contradiction(
        content_a: str, content_b: str, concept_a: str, concept_b: str
    ) -> dict:
        """Detect if two facts about the same concept contain contradictory numbers.

        Simple heuristic: if two facts share a concept and contain different
        numbers for what appears to be the same measurement, flag as
        contradiction.

        Args:
            content_a: Content of first fact.
            content_b: Content of second fact.
            concept_a: Concept of first fact.
            concept_b: Concept of second fact.

        Returns:
            Dict with contradiction info, or empty dict if none found.
        """
        concept_words_a = set(concept_a.lower().split()) if concept_a else set()
        concept_words_b = set(concept_b.lower().split()) if concept_b else set()

        if not concept_words_a or not concept_words_b:
            return {}

        common = concept_words_a & concept_words_b
        common = {w for w in common if len(w) > 2}
        if not common:
            return {}

        nums_a = re.findall(r"\b\d+(?:\.\d+)?\b", content_a)
        nums_b = re.findall(r"\b\d+(?:\.\d+)?\b", content_b)

        if not nums_a or not nums_b:
            return {}

        nums_a_set = set(nums_a)
        nums_b_set = set(nums_b)
        unique_to_a = nums_a_set - nums_b_set
        unique_to_b = nums_b_set - nums_a_set

        if unique_to_a and unique_to_b:
            return {
                "contradiction": True,
                "conflicting_values": (
                    f"{', '.join(sorted(unique_to_a))} vs "
                    f"{', '.join(sorted(unique_to_b))}"
                ),
            }

        return {}

    # ==================================================================
    # PROCEDURAL MEMORY
    # ==================================================================

    def store_procedure(
        self,
        name: str,
        steps: list[str],
        prerequisites: list[str] | None = None,
        source_id: str = "",
        tags: list[str] | None = None,
    ) -> str:
        """Store a reusable procedure.

        Args:
            name: Human-readable procedure name.
            steps: Ordered list of step descriptions.
            prerequisites: Conditions required before executing.
            source_id: Origin reference (unused in storage but kept for API compat).
            tags: Categorisation tags (unused in storage but kept for API compat).

        Returns:
            node_id of the stored procedure.
        """
        node_id = _new_id("proc")
        now = _ts_now()
        steps_json = json.dumps(steps)
        prereqs_json = json.dumps(prerequisites) if prerequisites else "[]"

        self._conn.execute(
            """
            CREATE (:ProceduralMemory {
                node_id: $nid,
                agent_id: $aid,
                name: $name,
                steps: $steps,
                prerequisites: $prereqs,
                usage_count: 0,
                created_at: $ts
            })
            """,
            {
                "nid": node_id,
                "aid": self.agent_name,
                "name": name,
                "steps": steps_json,
                "prereqs": prereqs_json,
                "ts": now,
            },
        )
        return node_id

    def recall_procedure(
        self,
        query: str,
        limit: int = 5,
    ) -> list[ProceduralMemory]:
        """Recall procedures matching a query and increment their usage count.

        Args:
            query: Search keywords (matched against name and steps).
            limit: Maximum results.

        Returns:
            List of matching ProceduralMemory sorted by usage_count desc.
        """
        keywords = [w.strip() for w in query.split() if w.strip()]
        if not keywords:
            # Return most-used procedures
            result = self._conn.execute(
                """
                MATCH (p:ProceduralMemory)
                WHERE p.agent_id = $aid
                RETURN p.node_id, p.name, p.steps,
                       p.prerequisites, p.usage_count, p.created_at
                ORDER BY p.usage_count DESC
                LIMIT $lim
                """,
                {"aid": self.agent_name, "lim": limit},
            )
        else:
            conditions = []
            params: dict = {"aid": self.agent_name, "lim": limit}
            for i, kw in enumerate(keywords[:6]):
                pname = f"kw{i}"
                conditions.append(
                    f"(lower(p.name) CONTAINS lower(${pname}) "
                    f"OR lower(p.steps) CONTAINS lower(${pname}))"
                )
                params[pname] = kw

            where_kw = " OR ".join(conditions)
            result = self._conn.execute(
                f"""
                MATCH (p:ProceduralMemory)
                WHERE p.agent_id = $aid AND ({where_kw})
                RETURN p.node_id, p.name, p.steps,
                       p.prerequisites, p.usage_count, p.created_at
                ORDER BY p.usage_count DESC
                LIMIT $lim
                """,
                params,
            )

        procs: list[ProceduralMemory] = []
        while result.has_next():
            row = result.get_next()
            steps: list[str] = []
            if row[2]:
                try:
                    steps = json.loads(row[2])
                except (json.JSONDecodeError, TypeError):
                    pass
            prereqs: list[str] = []
            if row[3]:
                try:
                    prereqs = json.loads(row[3])
                except (json.JSONDecodeError, TypeError):
                    pass
            procs.append(ProceduralMemory(
                node_id=row[0],
                name=row[1],
                steps=steps,
                prerequisites=prereqs,
                usage_count=int(row[4]),
                created_at=datetime.fromtimestamp(row[5]),
            ))

        # Increment usage_count for all recalled procedures
        for proc in procs:
            self._conn.execute(
                """
                MATCH (p:ProceduralMemory)
                WHERE p.node_id = $nid
                SET p.usage_count = p.usage_count + 1
                """,
                {"nid": proc.node_id},
            )

        return procs

    # ==================================================================
    # PROSPECTIVE MEMORY
    # ==================================================================

    def store_prospective(
        self,
        description: str,
        trigger_condition: str,
        action_on_trigger: str,
        priority: int = 1,
    ) -> str:
        """Store a trigger-action pair for future evaluation.

        Args:
            description: What this prospective memory is about.
            trigger_condition: Text description of the trigger.
            action_on_trigger: What to do when fired.
            priority: Priority level (higher = more important).

        Returns:
            node_id of the stored prospective memory.
        """
        node_id = _new_id("pro")
        now = _ts_now()

        self._conn.execute(
            """
            CREATE (:ProspectiveMemory {
                node_id: $nid,
                agent_id: $aid,
                desc_text: $dtxt,
                trigger_condition: $trig,
                action_on_trigger: $act,
                status: $stat,
                priority: $pri,
                created_at: $ts
            })
            """,
            {
                "nid": node_id,
                "aid": self.agent_name,
                "dtxt": description,
                "trig": trigger_condition,
                "act": action_on_trigger,
                "stat": "pending",
                "pri": priority,
                "ts": now,
            },
        )
        return node_id

    def check_triggers(self, content: str) -> list[ProspectiveMemory]:
        """Check pending prospective memories against provided content.

        A simple keyword-overlap heuristic: if any word from the
        trigger_condition appears in ``content``, the prospective memory
        is considered triggered.

        Args:
            content: Text to evaluate against triggers.

        Returns:
            List of triggered ProspectiveMemory items (status set to "triggered").
        """
        result = self._conn.execute(
            """
            MATCH (p:ProspectiveMemory)
            WHERE p.agent_id = $aid AND p.status = $stat
            RETURN p.node_id, p.desc_text, p.trigger_condition,
                   p.action_on_trigger, p.status, p.priority, p.created_at
            ORDER BY p.priority DESC
            """,
            {"aid": self.agent_name, "stat": "pending"},
        )

        candidates: list[ProspectiveMemory] = []
        while result.has_next():
            row = result.get_next()
            candidates.append(ProspectiveMemory(
                node_id=row[0],
                description=row[1],
                trigger_condition=row[2],
                action_on_trigger=row[3],
                status=row[4],
                priority=int(row[5]),
                created_at=datetime.fromtimestamp(row[6]),
            ))

        content_lower = content.lower()
        triggered: list[ProspectiveMemory] = []
        for pm in candidates:
            trigger_words = {w.lower() for w in pm.trigger_condition.split() if w.strip()}
            if any(w in content_lower for w in trigger_words):
                # Mark as triggered in the DB
                self._conn.execute(
                    """
                    MATCH (p:ProspectiveMemory)
                    WHERE p.node_id = $nid
                    SET p.status = $stat
                    """,
                    {"nid": pm.node_id, "stat": "triggered"},
                )
                pm.status = "triggered"
                triggered.append(pm)

        return triggered

    def resolve_prospective(self, node_id: str) -> None:
        """Mark a prospective memory as resolved.

        Args:
            node_id: The node_id to resolve.
        """
        self._conn.execute(
            """
            MATCH (p:ProspectiveMemory)
            WHERE p.node_id = $nid AND p.agent_id = $aid
            SET p.status = $stat
            """,
            {"nid": node_id, "aid": self.agent_name, "stat": "resolved"},
        )

    # ==================================================================
    # STATISTICS
    # ==================================================================

    def get_statistics(self) -> dict:
        """Return counts per memory type.

        Returns:
            Dictionary with keys for each MemoryCategory and their counts,
            plus a ``total`` key.
        """
        tables = {
            MemoryCategory.SENSORY: "SensoryMemory",
            MemoryCategory.WORKING: "WorkingMemory",
            MemoryCategory.EPISODIC: "EpisodicMemory",
            MemoryCategory.SEMANTIC: "SemanticMemory",
            MemoryCategory.PROCEDURAL: "ProceduralMemory",
            MemoryCategory.PROSPECTIVE: "ProspectiveMemory",
        }

        stats: dict = {}
        total = 0
        for category, table in tables.items():
            result = self._conn.execute(
                f"MATCH (n:{table}) WHERE n.agent_id = $aid RETURN count(n)",
                {"aid": self.agent_name},
            )
            count = 0
            if result.has_next():
                count = int(result.get_next()[0])
            stats[category.value] = count
            total += count

        stats["total"] = total
        return stats

    # ==================================================================
    # EXPORT / IMPORT
    # ==================================================================

    def export_to_json(self) -> dict[str, Any]:
        """Export all memory nodes and edges to a JSON-serializable dict.

        Queries all node tables (SemanticMemory, EpisodicMemory, ProceduralMemory,
        ProspectiveMemory, SensoryMemory, WorkingMemory, ConsolidatedEpisode) and
        all edge types (SIMILAR_TO, DERIVES_FROM, SUPERSEDES, CONSOLIDATES,
        ATTENDED_TO, PROCEDURE_DERIVES_FROM) for this agent.

        Returns a portable dict that can be serialized to JSON and imported into
        another CognitiveMemory instance via ``import_from_json()``.

        Returns:
            Dict with keys for each node type, each edge type, metadata, and
            statistics.
        """
        export_data: dict[str, Any] = {
            "agent_name": self.agent_name,
            "exported_at": datetime.now().isoformat(),
            "format_version": "1.0",
            "semantic_nodes": [],
            "episodic_nodes": [],
            "procedural_nodes": [],
            "prospective_nodes": [],
            "sensory_nodes": [],
            "working_nodes": [],
            "consolidated_nodes": [],
            "similar_to_edges": [],
            "derives_from_edges": [],
            "supersedes_edges": [],
            "consolidates_edges": [],
            "attended_to_edges": [],
            "procedure_derives_from_edges": [],
            "statistics": {},
        }

        # -- Semantic nodes --
        try:
            result = self._conn.execute(
                """
                MATCH (s:SemanticMemory)
                WHERE s.agent_id = $aid
                RETURN s.node_id, s.concept, s.content, s.confidence,
                       s.source_id, s.tags, s.metadata, s.created_at,
                       s.entity_name
                ORDER BY s.created_at ASC
                """,
                {"aid": self.agent_name},
            )
            while result.has_next():
                row = result.get_next()
                export_data["semantic_nodes"].append({
                    "node_id": row[0],
                    "concept": row[1],
                    "content": row[2],
                    "confidence": row[3],
                    "source_id": row[4] or "",
                    "tags": json.loads(row[5]) if row[5] else [],
                    "metadata": json.loads(row[6]) if row[6] else {},
                    "created_at": row[7],
                    "entity_name": row[8] or "",
                })
        except Exception:
            pass

        # -- Episodic nodes --
        try:
            result = self._conn.execute(
                """
                MATCH (e:EpisodicMemory)
                WHERE e.agent_id = $aid
                RETURN e.node_id, e.content, e.source_label,
                       e.temporal_index, e.compressed, e.metadata,
                       e.created_at
                ORDER BY e.temporal_index ASC
                """,
                {"aid": self.agent_name},
            )
            while result.has_next():
                row = result.get_next()
                export_data["episodic_nodes"].append({
                    "node_id": row[0],
                    "content": row[1],
                    "source_label": row[2] or "",
                    "temporal_index": row[3],
                    "compressed": bool(row[4]),
                    "metadata": json.loads(row[5]) if row[5] else {},
                    "created_at": row[6],
                })
        except Exception:
            pass

        # -- Procedural nodes --
        try:
            result = self._conn.execute(
                """
                MATCH (p:ProceduralMemory)
                WHERE p.agent_id = $aid
                RETURN p.node_id, p.name, p.steps, p.prerequisites,
                       p.usage_count, p.created_at
                ORDER BY p.created_at ASC
                """,
                {"aid": self.agent_name},
            )
            while result.has_next():
                row = result.get_next()
                export_data["procedural_nodes"].append({
                    "node_id": row[0],
                    "name": row[1],
                    "steps": json.loads(row[2]) if row[2] else [],
                    "prerequisites": json.loads(row[3]) if row[3] else [],
                    "usage_count": row[4],
                    "created_at": row[5],
                })
        except Exception:
            pass

        # -- Prospective nodes --
        try:
            result = self._conn.execute(
                """
                MATCH (p:ProspectiveMemory)
                WHERE p.agent_id = $aid
                RETURN p.node_id, p.desc_text, p.trigger_condition,
                       p.action_on_trigger, p.status, p.priority,
                       p.created_at
                ORDER BY p.created_at ASC
                """,
                {"aid": self.agent_name},
            )
            while result.has_next():
                row = result.get_next()
                export_data["prospective_nodes"].append({
                    "node_id": row[0],
                    "description": row[1],
                    "trigger_condition": row[2],
                    "action_on_trigger": row[3],
                    "status": row[4],
                    "priority": row[5],
                    "created_at": row[6],
                })
        except Exception:
            pass

        # -- Sensory nodes --
        try:
            result = self._conn.execute(
                """
                MATCH (s:SensoryMemory)
                WHERE s.agent_id = $aid
                RETURN s.node_id, s.modality, s.raw_data,
                       s.observation_order, s.expires_at, s.created_at
                ORDER BY s.observation_order ASC
                """,
                {"aid": self.agent_name},
            )
            while result.has_next():
                row = result.get_next()
                export_data["sensory_nodes"].append({
                    "node_id": row[0],
                    "modality": row[1],
                    "raw_data": row[2],
                    "observation_order": row[3],
                    "expires_at": row[4],
                    "created_at": row[5],
                })
        except Exception:
            pass

        # -- Working memory nodes --
        try:
            result = self._conn.execute(
                """
                MATCH (w:WorkingMemory)
                WHERE w.agent_id = $aid
                RETURN w.node_id, w.slot_type, w.content,
                       w.relevance, w.task_id, w.created_at
                ORDER BY w.created_at ASC
                """,
                {"aid": self.agent_name},
            )
            while result.has_next():
                row = result.get_next()
                export_data["working_nodes"].append({
                    "node_id": row[0],
                    "slot_type": row[1],
                    "content": row[2],
                    "relevance": row[3],
                    "task_id": row[4],
                    "created_at": row[5],
                })
        except Exception:
            pass

        # -- Consolidated episode nodes --
        try:
            result = self._conn.execute(
                """
                MATCH (c:ConsolidatedEpisode)
                WHERE c.agent_id = $aid
                RETURN c.node_id, c.summary, c.original_count, c.created_at
                ORDER BY c.created_at ASC
                """,
                {"aid": self.agent_name},
            )
            while result.has_next():
                row = result.get_next()
                export_data["consolidated_nodes"].append({
                    "node_id": row[0],
                    "summary": row[1],
                    "original_count": row[2],
                    "created_at": row[3],
                })
        except Exception:
            pass

        # -- SIMILAR_TO edges --
        try:
            result = self._conn.execute(
                """
                MATCH (a:SemanticMemory)-[r:SIMILAR_TO]->(b:SemanticMemory)
                WHERE a.agent_id = $aid
                RETURN a.node_id, b.node_id, r.similarity_score
                """,
                {"aid": self.agent_name},
            )
            while result.has_next():
                row = result.get_next()
                export_data["similar_to_edges"].append({
                    "source_id": row[0],
                    "target_id": row[1],
                    "similarity_score": row[2],
                })
        except Exception:
            pass

        # -- DERIVES_FROM edges --
        try:
            result = self._conn.execute(
                """
                MATCH (s:SemanticMemory)-[r:DERIVES_FROM]->(e:EpisodicMemory)
                WHERE s.agent_id = $aid
                RETURN s.node_id, e.node_id, r.derived_at
                """,
                {"aid": self.agent_name},
            )
            while result.has_next():
                row = result.get_next()
                export_data["derives_from_edges"].append({
                    "source_id": row[0],
                    "target_id": row[1],
                    "derived_at": row[2],
                })
        except Exception:
            pass

        # -- SUPERSEDES edges --
        try:
            result = self._conn.execute(
                """
                MATCH (newer:SemanticMemory)-[r:SUPERSEDES]->(older:SemanticMemory)
                WHERE newer.agent_id = $aid
                RETURN newer.node_id, older.node_id, r.reason, r.temporal_delta
                """,
                {"aid": self.agent_name},
            )
            while result.has_next():
                row = result.get_next()
                export_data["supersedes_edges"].append({
                    "source_id": row[0],
                    "target_id": row[1],
                    "reason": row[2] or "",
                    "temporal_delta": row[3] or "",
                })
        except Exception:
            pass

        # -- CONSOLIDATES edges --
        try:
            result = self._conn.execute(
                """
                MATCH (c:ConsolidatedEpisode)-[r:CONSOLIDATES]->(e:EpisodicMemory)
                WHERE c.agent_id = $aid
                RETURN c.node_id, e.node_id, r.consolidated_at
                """,
                {"aid": self.agent_name},
            )
            while result.has_next():
                row = result.get_next()
                export_data["consolidates_edges"].append({
                    "source_id": row[0],
                    "target_id": row[1],
                    "consolidated_at": row[2],
                })
        except Exception:
            pass

        # -- ATTENDED_TO edges --
        try:
            result = self._conn.execute(
                """
                MATCH (s:SensoryMemory)-[r:ATTENDED_TO]->(e:EpisodicMemory)
                WHERE s.agent_id = $aid
                RETURN s.node_id, e.node_id, r.attended_at
                """,
                {"aid": self.agent_name},
            )
            while result.has_next():
                row = result.get_next()
                export_data["attended_to_edges"].append({
                    "source_id": row[0],
                    "target_id": row[1],
                    "attended_at": row[2],
                })
        except Exception:
            pass

        # -- PROCEDURE_DERIVES_FROM edges --
        try:
            result = self._conn.execute(
                """
                MATCH (p:ProceduralMemory)-[r:PROCEDURE_DERIVES_FROM]->(e:EpisodicMemory)
                WHERE p.agent_id = $aid
                RETURN p.node_id, e.node_id, r.derived_at
                """,
                {"aid": self.agent_name},
            )
            while result.has_next():
                row = result.get_next()
                export_data["procedure_derives_from_edges"].append({
                    "source_id": row[0],
                    "target_id": row[1],
                    "derived_at": row[2],
                })
        except Exception:
            pass

        # Compute statistics
        export_data["statistics"] = {
            "semantic_count": len(export_data["semantic_nodes"]),
            "episodic_count": len(export_data["episodic_nodes"]),
            "procedural_count": len(export_data["procedural_nodes"]),
            "prospective_count": len(export_data["prospective_nodes"]),
            "sensory_count": len(export_data["sensory_nodes"]),
            "working_count": len(export_data["working_nodes"]),
            "consolidated_count": len(export_data["consolidated_nodes"]),
            "total_nodes": sum(
                len(export_data[k])
                for k in [
                    "semantic_nodes", "episodic_nodes", "procedural_nodes",
                    "prospective_nodes", "sensory_nodes", "working_nodes",
                    "consolidated_nodes",
                ]
            ),
            "total_edges": sum(
                len(export_data[k])
                for k in [
                    "similar_to_edges", "derives_from_edges",
                    "supersedes_edges", "consolidates_edges",
                    "attended_to_edges", "procedure_derives_from_edges",
                ]
            ),
        }

        return export_data

    def import_from_json(
        self,
        data: dict[str, Any],
        merge: bool = False,
    ) -> dict[str, Any]:
        """Import memory from a JSON-serializable dict into this agent's graph.

        Reconstructs all node types and edge types from a previously exported
        dict (as produced by ``export_to_json()``).

        Args:
            data: Dict matching the format produced by ``export_to_json()``.
            merge: If True, adds imported nodes to existing memory (skipping
                   duplicates). If False, clears existing memory first.

        Returns:
            Dict with import statistics:
                - nodes_imported: Total nodes created
                - edges_imported: Total edges created
                - skipped: Count of nodes skipped (already exist in merge mode)
                - errors: Count of individual import errors
        """
        stats: dict[str, Any] = {
            "nodes_imported": 0,
            "edges_imported": 0,
            "skipped": 0,
            "errors": 0,
        }

        # Validate format version
        fmt_version = data.get("format_version", "")
        if fmt_version and fmt_version != "1.0":
            pass  # attempt import anyway

        # Clear existing data if not merging
        if not merge:
            self._clear_agent_data()

        # Build set of existing node IDs for merge dedup
        existing_ids: set[str] = set()
        if merge:
            existing_ids = self._get_existing_node_ids()

        # Helper for safe node import
        def _import_node(query: str, params: dict, node_id: str) -> bool:
            if node_id in existing_ids:
                stats["skipped"] += 1
                return False
            try:
                self._conn.execute(query, params)
                stats["nodes_imported"] += 1
                return True
            except Exception:
                stats["errors"] += 1
                return False

        def _import_edge(query: str, params: dict) -> bool:
            try:
                self._conn.execute(query, params)
                stats["edges_imported"] += 1
                return True
            except Exception:
                stats["errors"] += 1
                return False

        # -- Import EpisodicMemory nodes first (referenced by edges) --
        for node in data.get("episodic_nodes", []):
            nid = node.get("node_id", "")
            if not nid:
                stats["errors"] += 1
                continue
            _import_node(
                """
                CREATE (:EpisodicMemory {
                    node_id: $nid, agent_id: $aid, content: $cnt,
                    source_label: $src, temporal_index: $tidx,
                    compressed: $comp, metadata: $meta, created_at: $ts
                })
                """,
                {
                    "nid": nid,
                    "aid": self.agent_name,
                    "cnt": node.get("content", ""),
                    "src": node.get("source_label", ""),
                    "tidx": node.get("temporal_index", 0),
                    "comp": node.get("compressed", False),
                    "meta": json.dumps(node.get("metadata", {})),
                    "ts": node.get("created_at", 0),
                },
                nid,
            )

        # -- Import SemanticMemory nodes --
        for node in data.get("semantic_nodes", []):
            nid = node.get("node_id", "")
            if not nid:
                stats["errors"] += 1
                continue
            _import_node(
                """
                CREATE (:SemanticMemory {
                    node_id: $nid, agent_id: $aid, concept: $con,
                    content: $cnt, confidence: $conf, source_id: $src,
                    tags: $tags, metadata: $meta, created_at: $ts,
                    entity_name: $ename
                })
                """,
                {
                    "nid": nid,
                    "aid": self.agent_name,
                    "con": node.get("concept", ""),
                    "cnt": node.get("content", ""),
                    "conf": node.get("confidence", 0.8),
                    "src": node.get("source_id", ""),
                    "tags": json.dumps(node.get("tags", [])),
                    "meta": json.dumps(node.get("metadata", {})),
                    "ts": node.get("created_at", 0),
                    "ename": node.get("entity_name", ""),
                },
                nid,
            )

        # -- Import ProceduralMemory nodes --
        for node in data.get("procedural_nodes", []):
            nid = node.get("node_id", "")
            if not nid:
                stats["errors"] += 1
                continue
            _import_node(
                """
                CREATE (:ProceduralMemory {
                    node_id: $nid, agent_id: $aid, name: $name,
                    steps: $steps, prerequisites: $prereqs,
                    usage_count: $uc, created_at: $ts
                })
                """,
                {
                    "nid": nid,
                    "aid": self.agent_name,
                    "name": node.get("name", ""),
                    "steps": json.dumps(node.get("steps", [])),
                    "prereqs": json.dumps(node.get("prerequisites", [])),
                    "uc": node.get("usage_count", 0),
                    "ts": node.get("created_at", 0),
                },
                nid,
            )

        # -- Import ProspectiveMemory nodes --
        for node in data.get("prospective_nodes", []):
            nid = node.get("node_id", "")
            if not nid:
                stats["errors"] += 1
                continue
            _import_node(
                """
                CREATE (:ProspectiveMemory {
                    node_id: $nid, agent_id: $aid, desc_text: $dtxt,
                    trigger_condition: $trig, action_on_trigger: $act,
                    status: $stat, priority: $pri, created_at: $ts
                })
                """,
                {
                    "nid": nid,
                    "aid": self.agent_name,
                    "dtxt": node.get("description", ""),
                    "trig": node.get("trigger_condition", ""),
                    "act": node.get("action_on_trigger", ""),
                    "stat": node.get("status", "pending"),
                    "pri": node.get("priority", 1),
                    "ts": node.get("created_at", 0),
                },
                nid,
            )

        # -- Import SensoryMemory nodes --
        for node in data.get("sensory_nodes", []):
            nid = node.get("node_id", "")
            if not nid:
                stats["errors"] += 1
                continue
            _import_node(
                """
                CREATE (:SensoryMemory {
                    node_id: $nid, agent_id: $aid, modality: $mod,
                    raw_data: $data, observation_order: $ord,
                    expires_at: $exp, created_at: $ts
                })
                """,
                {
                    "nid": nid,
                    "aid": self.agent_name,
                    "mod": node.get("modality", ""),
                    "data": node.get("raw_data", ""),
                    "ord": node.get("observation_order", 0),
                    "exp": node.get("expires_at", 0),
                    "ts": node.get("created_at", 0),
                },
                nid,
            )

        # -- Import WorkingMemory nodes --
        for node in data.get("working_nodes", []):
            nid = node.get("node_id", "")
            if not nid:
                stats["errors"] += 1
                continue
            _import_node(
                """
                CREATE (:WorkingMemory {
                    node_id: $nid, agent_id: $aid, slot_type: $st,
                    content: $cnt, relevance: $rel,
                    task_id: $tid, created_at: $ts
                })
                """,
                {
                    "nid": nid,
                    "aid": self.agent_name,
                    "st": node.get("slot_type", ""),
                    "cnt": node.get("content", ""),
                    "rel": node.get("relevance", 1.0),
                    "tid": node.get("task_id", ""),
                    "ts": node.get("created_at", 0),
                },
                nid,
            )

        # -- Import ConsolidatedEpisode nodes --
        for node in data.get("consolidated_nodes", []):
            nid = node.get("node_id", "")
            if not nid:
                stats["errors"] += 1
                continue
            _import_node(
                """
                CREATE (:ConsolidatedEpisode {
                    node_id: $nid, agent_id: $aid, summary: $sum,
                    original_count: $cnt, created_at: $ts
                })
                """,
                {
                    "nid": nid,
                    "aid": self.agent_name,
                    "sum": node.get("summary", ""),
                    "cnt": node.get("original_count", 0),
                    "ts": node.get("created_at", 0),
                },
                nid,
            )

        # -- Import edges --
        for edge in data.get("similar_to_edges", []):
            _import_edge(
                """
                MATCH (a:SemanticMemory {node_id: $sid})
                MATCH (b:SemanticMemory {node_id: $tid})
                CREATE (a)-[:SIMILAR_TO {similarity_score: $score}]->(b)
                """,
                {
                    "sid": edge["source_id"],
                    "tid": edge["target_id"],
                    "score": edge.get("similarity_score", 0.0),
                },
            )

        for edge in data.get("derives_from_edges", []):
            _import_edge(
                """
                MATCH (s:SemanticMemory {node_id: $sid})
                MATCH (e:EpisodicMemory {node_id: $tid})
                CREATE (s)-[:DERIVES_FROM {derived_at: $ts}]->(e)
                """,
                {
                    "sid": edge["source_id"],
                    "tid": edge["target_id"],
                    "ts": edge.get("derived_at", 0),
                },
            )

        for edge in data.get("supersedes_edges", []):
            _import_edge(
                """
                MATCH (newer:SemanticMemory {node_id: $sid})
                MATCH (older:SemanticMemory {node_id: $tid})
                CREATE (newer)-[:SUPERSEDES {
                    reason: $reason, temporal_delta: $delta
                }]->(older)
                """,
                {
                    "sid": edge["source_id"],
                    "tid": edge["target_id"],
                    "reason": edge.get("reason", ""),
                    "delta": edge.get("temporal_delta", ""),
                },
            )

        for edge in data.get("consolidates_edges", []):
            _import_edge(
                """
                MATCH (c:ConsolidatedEpisode {node_id: $sid})
                MATCH (e:EpisodicMemory {node_id: $tid})
                CREATE (c)-[:CONSOLIDATES {consolidated_at: $ts}]->(e)
                """,
                {
                    "sid": edge["source_id"],
                    "tid": edge["target_id"],
                    "ts": edge.get("consolidated_at", 0),
                },
            )

        for edge in data.get("attended_to_edges", []):
            _import_edge(
                """
                MATCH (s:SensoryMemory {node_id: $sid})
                MATCH (e:EpisodicMemory {node_id: $tid})
                CREATE (s)-[:ATTENDED_TO {attended_at: $ts}]->(e)
                """,
                {
                    "sid": edge["source_id"],
                    "tid": edge["target_id"],
                    "ts": edge.get("attended_at", 0),
                },
            )

        for edge in data.get("procedure_derives_from_edges", []):
            _import_edge(
                """
                MATCH (p:ProceduralMemory {node_id: $sid})
                MATCH (e:EpisodicMemory {node_id: $tid})
                CREATE (p)-[:PROCEDURE_DERIVES_FROM {derived_at: $ts}]->(e)
                """,
                {
                    "sid": edge["source_id"],
                    "tid": edge["target_id"],
                    "ts": edge.get("derived_at", 0),
                },
            )

        # Update internal monotonic counters after import
        self._sensory_order = self._load_max_order("SensoryMemory", "observation_order")
        self._temporal_index = self._load_max_order("EpisodicMemory", "temporal_index")

        return stats

    def _clear_agent_data(self) -> None:
        """Delete all nodes and edges belonging to this agent.

        Used by ``import_from_json`` when ``merge=False`` to start fresh.
        Deletes edges first (Kuzu requires this before node deletion),
        then deletes all nodes.
        """
        # Delete edges referencing this agent's nodes
        edge_queries = [
            "MATCH (a:SemanticMemory {agent_id: $aid})-[r:SIMILAR_TO]->() DELETE r",
            "MATCH ()-[r:SIMILAR_TO]->(b:SemanticMemory {agent_id: $aid}) DELETE r",
            "MATCH (s:SemanticMemory {agent_id: $aid})-[r:DERIVES_FROM]->() DELETE r",
            "MATCH (n:SemanticMemory {agent_id: $aid})-[r:SUPERSEDES]->() DELETE r",
            "MATCH ()-[r:SUPERSEDES]->(o:SemanticMemory {agent_id: $aid}) DELETE r",
            "MATCH (c:ConsolidatedEpisode {agent_id: $aid})-[r:CONSOLIDATES]->() DELETE r",
            "MATCH (s:SensoryMemory {agent_id: $aid})-[r:ATTENDED_TO]->() DELETE r",
            "MATCH (p:ProceduralMemory {agent_id: $aid})-[r:PROCEDURE_DERIVES_FROM]->() DELETE r",
        ]
        for query in edge_queries:
            try:
                self._conn.execute(query, {"aid": self.agent_name})
            except Exception:
                pass  # edge table may not exist yet

        # Delete nodes from all tables
        node_tables = [
            "SemanticMemory", "EpisodicMemory", "ProceduralMemory",
            "ProspectiveMemory", "SensoryMemory", "WorkingMemory",
            "ConsolidatedEpisode",
        ]
        for table in node_tables:
            try:
                self._conn.execute(
                    f"MATCH (n:{table} {{agent_id: $aid}}) DELETE n",
                    {"aid": self.agent_name},
                )
            except Exception:
                pass  # table may not exist yet

    def _get_existing_node_ids(self) -> set[str]:
        """Get all existing node IDs for this agent (for merge dedup).

        Returns:
            Set of node_id strings across all node tables.
        """
        ids: set[str] = set()
        tables = [
            "SemanticMemory", "EpisodicMemory", "ProceduralMemory",
            "ProspectiveMemory", "SensoryMemory", "WorkingMemory",
            "ConsolidatedEpisode",
        ]
        for table in tables:
            try:
                result = self._conn.execute(
                    f"MATCH (n:{table} {{agent_id: $aid}}) RETURN n.node_id",
                    {"aid": self.agent_name},
                )
                while result.has_next():
                    ids.add(result.get_next()[0])
            except Exception:
                pass
        return ids

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release the Kuzu connection."""
        self._conn = None
        self._db = None
