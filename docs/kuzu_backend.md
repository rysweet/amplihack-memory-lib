# Kuzu Backend

amplihack-memory-lib uses [Kuzu](https://kuzudb.com/) as its primary graph database backend. Kuzu is an embedded graph database (like SQLite for graphs) that requires no server -- it runs in-process and stores data in a local directory.

---

## Why Kuzu

| Feature | Benefit |
|---------|---------|
| **Embedded** | No server setup, runs in-process |
| **Graph-native** | First-class support for nodes, edges, and graph traversal |
| **Cypher-compatible** | Uses openCypher query language |
| **Persistent** | Stores data on disk, survives process restarts |
| **Fast** | Columnar storage with vectorized execution |
| **Python bindings** | Native Python API via `kuzu` package |

---

## Database Layout

Each memory system creates a Kuzu database directory:

```
~/.amplihack/
    hierarchical_memory/
        my-agent/           # HierarchicalMemory db
            kuzu_db/        # Kuzu data directory
    memory/
        my-agent/           # MemoryConnector db
            kuzu_db/        # Kuzu data directory
```

CognitiveMemory also creates a Kuzu database at the specified path:

```
/path/to/cognitive_db/      # CognitiveMemory db (user-specified)
```

---

## Schema: HierarchicalMemory

### Node Tables

**SemanticMemory** -- Distilled facts:

```cypher
CREATE NODE TABLE SemanticMemory(
    memory_id STRING,        -- UUID primary key
    concept STRING,          -- Topic/concept label
    content STRING,          -- Factual content
    confidence DOUBLE,       -- 0.0-1.0
    source_id STRING,        -- Provenance (episode ID)
    agent_id STRING,         -- Agent isolation
    tags STRING,             -- JSON array of tags
    metadata STRING,         -- JSON object
    created_at STRING,       -- ISO timestamp
    entity_name STRING,      -- Extracted entity (lowercase)
    PRIMARY KEY (memory_id)
)
```

**EpisodicMemory** -- Raw source content:

```cypher
CREATE NODE TABLE EpisodicMemory(
    memory_id STRING,        -- UUID primary key
    content STRING,          -- Episode content
    source_label STRING,     -- Origin label
    agent_id STRING,         -- Agent isolation
    tags STRING,             -- JSON array
    metadata STRING,         -- JSON object
    created_at STRING,       -- ISO timestamp
    PRIMARY KEY (memory_id)
)
```

### Relationship Tables

```cypher
-- Text similarity between facts
CREATE REL TABLE SIMILAR_TO(
    FROM SemanticMemory TO SemanticMemory,
    weight DOUBLE,           -- Similarity score
    metadata STRING          -- JSON (contradiction info)
)

-- Fact derived from episode (provenance)
CREATE REL TABLE DERIVES_FROM(
    FROM SemanticMemory TO EpisodicMemory,
    extraction_method STRING,
    confidence DOUBLE
)

-- Newer fact supersedes older (temporal update)
CREATE REL TABLE SUPERSEDES(
    FROM SemanticMemory TO SemanticMemory,
    reason STRING,           -- Why it supersedes
    temporal_delta STRING    -- Time difference description
)

-- Explicit value transition chain
CREATE REL TABLE TRANSITIONED_TO(
    FROM SemanticMemory TO SemanticMemory,
    from_value STRING,       -- Newer value
    to_value STRING,         -- Older value
    turn INT64,              -- Temporal index
    transition_type STRING   -- e.g., "update"
)
```

---

## Schema: CognitiveMemory

### Node Tables

```cypher
CREATE NODE TABLE SensoryMemory(
    node_id STRING, agent_id STRING,
    modality STRING, raw_data STRING,
    observation_order INT64, expires_at INT64, created_at INT64,
    PRIMARY KEY(node_id)
)

CREATE NODE TABLE WorkingMemory(
    node_id STRING, agent_id STRING,
    slot_type STRING, content STRING,
    relevance DOUBLE, task_id STRING, created_at INT64,
    PRIMARY KEY(node_id)
)

CREATE NODE TABLE EpisodicMemory(
    node_id STRING, agent_id STRING,
    content STRING, source_label STRING,
    temporal_index INT64, compressed BOOLEAN,
    metadata STRING, created_at INT64,
    PRIMARY KEY(node_id)
)

CREATE NODE TABLE SemanticMemory(
    node_id STRING, agent_id STRING,
    concept STRING, content STRING,
    confidence DOUBLE, source_id STRING,
    tags STRING, metadata STRING, created_at INT64,
    PRIMARY KEY(node_id)
)

CREATE NODE TABLE ProceduralMemory(
    node_id STRING, agent_id STRING,
    name STRING, steps STRING,
    prerequisites STRING, usage_count INT64, created_at INT64,
    PRIMARY KEY(node_id)
)

CREATE NODE TABLE ProspectiveMemory(
    node_id STRING, agent_id STRING,
    desc_text STRING, trigger_condition STRING,
    action_on_trigger STRING, status STRING,
    priority INT64, created_at INT64,
    PRIMARY KEY(node_id)
)

CREATE NODE TABLE ConsolidatedEpisode(
    node_id STRING, agent_id STRING,
    summary STRING, original_count INT64, created_at INT64,
    PRIMARY KEY(node_id)
)
```

### Relationship Tables

```cypher
CREATE REL TABLE SIMILAR_TO(FROM SemanticMemory TO SemanticMemory, similarity_score DOUBLE)
CREATE REL TABLE DERIVES_FROM(FROM SemanticMemory TO EpisodicMemory, derived_at INT64)
CREATE REL TABLE PROCEDURE_DERIVES_FROM(FROM ProceduralMemory TO EpisodicMemory, derived_at INT64)
CREATE REL TABLE CONSOLIDATES(FROM ConsolidatedEpisode TO EpisodicMemory, consolidated_at INT64)
CREATE REL TABLE ATTENDED_TO(FROM SensoryMemory TO EpisodicMemory, attended_at INT64)
```

---

## Schema: KuzuBackend (Experience Store)

```cypher
CREATE NODE TABLE Experience(
    experience_id STRING,
    agent_name STRING,
    experience_type STRING,     -- success|failure|pattern|insight
    context STRING,
    outcome STRING,
    confidence DOUBLE,
    timestamp INT64,
    metadata STRING,            -- JSON
    tags STRING,                -- JSON array
    compressed BOOLEAN,
    PRIMARY KEY(experience_id)
)

CREATE REL TABLE SIMILAR_TO(
    FROM Experience TO Experience,
    similarity_score DOUBLE
)

CREATE REL TABLE LEADS_TO(
    FROM Experience TO Experience,
    causal_strength DOUBLE
)
```

---

## Query Patterns

### Keyword Search

```cypher
MATCH (m:SemanticMemory)
WHERE m.agent_id = $agent_id
  AND (lower(m.concept) CONTAINS lower($query)
       OR lower(m.content) CONTAINS lower($query))
RETURN m.memory_id, m.concept, m.content, m.confidence,
       m.source_id, m.tags, m.metadata, m.created_at, m.entity_name
ORDER BY m.confidence DESC
LIMIT $max_nodes
```

### Entity-Centric Retrieval

```cypher
MATCH (m:SemanticMemory)
WHERE m.agent_id = $agent_id
  AND m.entity_name = $entity_name
RETURN m.memory_id, m.concept, m.content, m.confidence,
       m.source_id, m.tags, m.metadata, m.created_at, m.entity_name
ORDER BY m.confidence DESC
LIMIT $max_nodes
```

### Similarity Traversal (Graph RAG)

```cypher
MATCH (m:SemanticMemory)-[e:SIMILAR_TO]-(n:SemanticMemory)
WHERE m.memory_id IN $seed_ids
  AND n.agent_id = $agent_id
  AND e.weight >= $threshold
RETURN n.memory_id, n.concept, n.content, n.confidence,
       n.source_id, n.tags, n.metadata, n.created_at, n.entity_name
```

### Supersedes Chain

```cypher
MATCH (newer:SemanticMemory)-[:SUPERSEDES]->(older:SemanticMemory)
WHERE newer.memory_id IN $seed_ids
  AND older.agent_id = $agent_id
RETURN older.memory_id, older.concept, older.content, older.confidence,
       older.source_id, older.tags, older.metadata, older.created_at,
       older.entity_name
```

### Transition Chain

```cypher
MATCH (a:SemanticMemory)-[t:TRANSITIONED_TO]->(b:SemanticMemory)
WHERE a.memory_id IN $node_ids OR b.memory_id IN $node_ids
RETURN a.memory_id, b.memory_id, t.from_value, t.to_value,
       t.turn, t.transition_type
```

---

## Performance Characteristics

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Store knowledge | O(N) | N = recent nodes for similarity comparison |
| Retrieve subgraph | O(K + E) | K = keyword matches, E = edges traversed |
| Entity lookup | O(1) amortized | Direct field match |
| Similarity edge creation | O(N * M) | N = new node tokens, M = existing node tokens |
| Export graph | O(V + E) | V = vertices, E = edges |

**Typical performance:**

- Store: ~5-20ms per node (including similarity edge computation)
- Retrieve: ~2-10ms for keyword + 1-hop traversal
- Scales well to thousands of nodes per agent

---

## SQLite Fallback

The `SQLiteBackend` provides a fallback for environments where Kuzu is not available:

```python
from amplihack_memory import MemoryConnector

# Use SQLite instead of Kuzu
connector = MemoryConnector(agent_name="my-agent", backend="sqlite")
```

Key differences:

| Feature | Kuzu | SQLite |
|---------|------|--------|
| Storage | Graph (nodes + edges) | Relational (tables) |
| Search | CONTAINS keyword match | FTS5 full-text search |
| Relationships | Native graph edges | Not available |
| Concurrency | Single connection | WAL mode + thread lock |
| Graph RAG | Full support | Not available |
| Use case | HierarchicalMemory | ExperienceStore only |

The SQLite backend is used by `MemoryConnector` and `ExperienceStore` only. The `HierarchicalMemory` and `CognitiveMemory` classes always require Kuzu.
