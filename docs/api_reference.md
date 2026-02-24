# API Reference

Complete API reference for all public classes and functions in `amplihack_memory`.

---

## HierarchicalMemory

```python
from amplihack_memory import HierarchicalMemory
```

### Constructor

```python
HierarchicalMemory(
    agent_name: str,
    db_path: str | Path | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agent_name` | `str` | required | Agent identifier (alphanumeric + hyphens/underscores, 1-64 chars) |
| `db_path` | `str \| Path \| None` | `None` | Path to Kuzu database directory. Defaults to `~/.amplihack/hierarchical_memory/<agent_name>` |

### Methods

#### store_knowledge

```python
store_knowledge(
    content: str,
    concept: str = "",
    confidence: float = 0.8,
    category: MemoryCategory | None = None,
    source_id: str = "",
    tags: list[str] | None = None,
    temporal_metadata: dict | None = None,
) -> str
```

Store a knowledge node. Returns the `node_id` (UUID string).

#### store_episode

```python
store_episode(content: str, source_label: str = "") -> str
```

Store an episodic memory node. Returns the `episode_id`.

#### retrieve_subgraph

```python
retrieve_subgraph(
    query: str,
    max_nodes: int = 20,
    similarity_threshold: float = 0.3,
) -> KnowledgeSubgraph
```

Retrieve a subgraph of related knowledge via Graph RAG.

#### get_all_knowledge

```python
get_all_knowledge(limit: int = 50) -> list[KnowledgeNode]
```

Get all knowledge nodes for this agent.

#### export_graph

```python
export_graph() -> dict
```

Export the entire graph as a serializable dict.

#### import_graph

```python
import_graph(data: dict) -> None
```

Import a graph from a previously exported dict.

#### store_fact (alias)

```python
store_fact(content: str, concept: str = "", confidence: float = 0.8, **kwargs) -> str
```

Protocol-compatible alias for `store_knowledge()`.

#### search_facts (alias)

```python
search_facts(query: str, max_nodes: int = 20, **kwargs) -> list[dict]
```

Protocol-compatible wrapper around `retrieve_subgraph()`. Returns list of dicts.

#### get_all_facts (alias)

```python
get_all_facts(limit: int = 50) -> list[dict]
```

Protocol-compatible wrapper around `get_all_knowledge()`. Returns list of dicts.

#### close

```python
close() -> None
```

Release the Kuzu database connection.

---

## CognitiveMemory

```python
from amplihack_memory import CognitiveMemory
```

### Constructor

```python
CognitiveMemory(
    agent_name: str,
    db_path: str | Path,
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `agent_name` | `str` | Agent identifier (cannot be empty) |
| `db_path` | `str \| Path` | Path to Kuzu database directory |

### Sensory Memory Methods

| Method | Signature | Returns |
|--------|-----------|---------|
| `record_sensory` | `(modality: str, raw_data: str, ttl_seconds: int = 300) -> str` | `node_id` |
| `get_recent_sensory` | `(limit: int = 10) -> list[SensoryItem]` | Non-expired items |
| `attend_to_sensory` | `(sensory_id: str, reason: str) -> str \| None` | Episodic `node_id` or `None` |
| `prune_expired_sensory` | `() -> int` | Count pruned |

### Working Memory Methods

| Method | Signature | Returns |
|--------|-----------|---------|
| `push_working` | `(slot_type: str, content: str, task_id: str, relevance: float = 1.0) -> str` | `node_id` |
| `get_working` | `(task_id: str) -> list[WorkingMemorySlot]` | Slots by relevance DESC |
| `clear_working` | `(task_id: str) -> int` | Count cleared |

### Episodic Memory Methods

| Method | Signature | Returns |
|--------|-----------|---------|
| `store_episode` | `(content: str, source_label: str, temporal_index: int \| None = None, metadata: dict \| None = None) -> str` | `node_id` |
| `get_episodes` | `(limit: int = 20, include_compressed: bool = False) -> list[EpisodicMemory]` | Episodes by temporal_index DESC |
| `consolidate_episodes` | `(batch_size: int = 10, summarizer: Callable \| None = None) -> str \| None` | ConsolidatedEpisode `node_id` or `None` |

### Semantic Memory Methods

| Method | Signature | Returns |
|--------|-----------|---------|
| `store_fact` | `(concept: str, content: str, confidence: float = 1.0, source_id: str = "", tags: list[str] \| None = None, temporal_metadata: dict \| None = None) -> str` | `node_id` |
| `search_facts` | `(query: str, limit: int = 10, min_confidence: float = 0.0) -> list[SemanticFact]` | Matching facts |
| `get_all_facts` | `(limit: int = 50) -> list[SemanticFact]` | All facts by confidence DESC |

### Procedural Memory Methods

| Method | Signature | Returns |
|--------|-----------|---------|
| `store_procedure` | `(name: str, steps: list[str], prerequisites: list[str] \| None = None, ...) -> str` | `node_id` |
| `recall_procedure` | `(query: str, limit: int = 5) -> list[ProceduralMemory]` | Matching procedures (usage_count incremented) |

### Prospective Memory Methods

| Method | Signature | Returns |
|--------|-----------|---------|
| `store_prospective` | `(description: str, trigger_condition: str, action_on_trigger: str, priority: int = 1) -> str` | `node_id` |
| `check_triggers` | `(content: str) -> list[ProspectiveMemory]` | Triggered items |
| `resolve_prospective` | `(node_id: str) -> None` | -- |

### Other Methods

| Method | Signature | Returns |
|--------|-----------|---------|
| `get_statistics` | `() -> dict` | Counts per memory type + total |
| `close` | `() -> None` | Release Kuzu connection |

---

## Data Classes

### KnowledgeNode

```python
from amplihack_memory import KnowledgeNode
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `node_id` | `str` | required | UUID identifier |
| `category` | `MemoryCategory` | required | Memory category |
| `content` | `str` | required | Main text content |
| `concept` | `str` | required | Topic/concept label |
| `confidence` | `float` | `0.8` | Confidence score |
| `source_id` | `str` | `""` | Provenance ID |
| `created_at` | `str` | `""` | ISO timestamp |
| `tags` | `list[str]` | `[]` | Tags |
| `metadata` | `dict` | `{}` | Additional metadata |

### KnowledgeEdge

```python
from amplihack_memory import KnowledgeEdge
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `source_id` | `str` | required | Source node ID |
| `target_id` | `str` | required | Target node ID |
| `relationship` | `str` | required | Edge type |
| `weight` | `float` | `1.0` | Edge weight |
| `metadata` | `dict` | `{}` | Additional metadata |

### KnowledgeSubgraph

```python
from amplihack_memory import KnowledgeSubgraph
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `nodes` | `list[KnowledgeNode]` | `[]` | Knowledge nodes |
| `edges` | `list[KnowledgeEdge]` | `[]` | Knowledge edges |
| `query` | `str` | `""` | Original query |

**Methods:**

- `to_llm_context(chronological: bool = False) -> str` -- Format as LLM-readable text

### MemoryCategory (HierarchicalMemory)

```python
from amplihack_memory.hierarchical_memory import MemoryCategory
```

| Value | String |
|-------|--------|
| `EPISODIC` | `"episodic"` |
| `SEMANTIC` | `"semantic"` |
| `PROCEDURAL` | `"procedural"` |
| `PROSPECTIVE` | `"prospective"` |
| `WORKING` | `"working"` |

### MemoryCategory (CognitiveMemory)

```python
from amplihack_memory import MemoryCategory
```

| Value | String |
|-------|--------|
| `SENSORY` | `"sensory"` |
| `WORKING` | `"working"` |
| `EPISODIC` | `"episodic"` |
| `SEMANTIC` | `"semantic"` |
| `PROCEDURAL` | `"procedural"` |
| `PROSPECTIVE` | `"prospective"` |

---

## Experience System

### Experience

```python
from amplihack_memory import Experience, ExperienceType
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `experience_type` | `ExperienceType` | required | Type enum |
| `context` | `str` | required | Situation (max 500 chars) |
| `outcome` | `str` | required | Result (max 1000 chars) |
| `confidence` | `float` | required | 0.0-1.0 |
| `timestamp` | `datetime` | `now()` | When it occurred |
| `experience_id` | `str` | auto-generated | `exp_YYYYMMDD_HHMMSS_hash` |
| `metadata` | `dict` | `{}` | Structured data |
| `tags` | `list[str]` | `[]` | Tags |

**Methods:**

- `to_dict() -> dict` -- Serialize to dictionary
- `from_dict(data) -> Experience` -- Class method to deserialize

### ExperienceType

| Value | String |
|-------|--------|
| `SUCCESS` | `"success"` |
| `FAILURE` | `"failure"` |
| `PATTERN` | `"pattern"` |
| `INSIGHT` | `"insight"` |

### ExperienceStore

```python
from amplihack_memory import ExperienceStore
```

```python
ExperienceStore(
    agent_name: str,
    auto_compress: bool = True,
    max_age_days: int | None = None,
    max_experiences: int | None = None,
    max_memory_mb: int = 100,
    storage_path: Path | None = None,
    backend: str = "kuzu",
)
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `add` | `(experience: Experience) -> str` | Add with auto-management |
| `search` | `(query, experience_type, min_confidence, limit) -> list[Experience]` | Full-text search |
| `get_statistics` | `() -> dict` | Storage stats |

### MemoryConnector

```python
from amplihack_memory import MemoryConnector
```

```python
MemoryConnector(
    agent_name: str,
    storage_path: Path | None = None,
    max_memory_mb: int = 100,
    enable_compression: bool = True,
    backend: str = "kuzu",
)
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `store_experience` | `(experience: Experience) -> str` | Store experience |
| `retrieve_experiences` | `(limit, experience_type, min_confidence) -> list[Experience]` | Retrieve |
| `search` | `(query, experience_type, min_confidence, limit) -> list[Experience]` | Search |
| `get_statistics` | `() -> dict` | Stats |
| `close` | `() -> None` | Close connection |

Supports context manager: `with MemoryConnector(...) as conn:`

---

## Shared Utilities

### Similarity

```python
from amplihack_memory import (
    compute_similarity,
    compute_word_similarity,
    compute_tag_similarity,
    rerank_facts_by_query,
)
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `compute_word_similarity` | `(text_a: str, text_b: str) -> float` | Jaccard on tokenized words |
| `compute_tag_similarity` | `(tags_a: list[str], tags_b: list[str]) -> float` | Jaccard on tag lists |
| `compute_similarity` | `(node_a: dict, node_b: dict) -> float` | Weighted composite (0.5w + 0.2t + 0.3c) |
| `rerank_facts_by_query` | `(facts: list[dict], query: str, top_k: int = 0) -> list[dict]` | Rerank by query relevance |

### Entity Extraction

```python
from amplihack_memory import extract_entity_name
```

```python
extract_entity_name(content: str, concept: str = "") -> str
```

Returns lowercase entity name or empty string.

### Contradiction Detection

```python
from amplihack_memory import detect_contradiction
```

```python
detect_contradiction(
    content_a: str, content_b: str,
    concept_a: str = "", concept_b: str = "",
) -> dict
```

Returns `{"contradiction": True, "conflicting_values": "..."}` or empty dict.

---

## Security

```python
from amplihack_memory import (
    AgentCapabilities,
    ScopeLevel,
    CredentialScrubber,
    QueryValidator,
    SecureMemoryBackend,
)
```

### ScopeLevel

| Value | Description |
|-------|-------------|
| `SESSION_ONLY` | Current session only |
| `CROSS_SESSION_READ` | Read from other sessions |
| `CROSS_SESSION_WRITE` | Write across sessions |
| `GLOBAL_READ` | Read all agents' memories |
| `GLOBAL_WRITE` | Write to any agent's memory |

### AgentCapabilities

```python
AgentCapabilities(
    scope: ScopeLevel,
    allowed_experience_types: list[ExperienceType],
    max_query_cost: int,
    can_access_patterns: bool,
    memory_quota_mb: int,
)
```

### CredentialScrubber

```python
scrubber = CredentialScrubber()
scrubbed_exp, was_scrubbed = scrubber.scrub_experience(experience)
has_creds = scrubber.contains_credentials(text)
```

### SecureMemoryBackend

```python
secure = SecureMemoryBackend(store, capabilities)
secure.add_experience(experience)  # Enforces caps + scrubs
secure.search(query)               # Enforces type restrictions
```

---

## Exceptions

```python
from amplihack_memory import (
    MemoryError,               # Base exception
    ExperienceNotFoundError,   # Experience not found
    InvalidExperienceError,    # Validation failure
    MemoryQuotaExceededError,  # Storage quota exceeded
    SecurityViolationError,    # Security policy violation
    QueryCostExceededError,    # Query too expensive
)
```
