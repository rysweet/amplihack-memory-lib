# Cognitive Memory

The `CognitiveMemory` class provides a six-type memory system modeled after human cognition, where each memory type has distinct storage, lifecycle, and retrieval characteristics.

---

## Overview

CognitiveMemory manages six separate Kuzu node tables -- one per memory type -- plus relationship edges that connect them:

```
                    ATTENDED_TO
    SensoryMemory -----------------> EpisodicMemory
                                          |
                                    DERIVES_FROM
                                          |
                                          v
                                    SemanticMemory
                                     |          |
                               SIMILAR_TO   SIMILAR_TO
                                     |          |
                                     v          v
                              SemanticMemory  SemanticMemory

    ProceduralMemory  <--- PROCEDURE_DERIVES_FROM --- EpisodicMemory

    ConsolidatedEpisode --- CONSOLIDATES ---> EpisodicMemory (compressed)

    ProspectiveMemory (standalone, checked against content triggers)
    WorkingMemory (standalone, bounded capacity per task)
```

---

## The Six Memory Types

### 1. Sensory Memory

**Purpose:** Short-lived raw observations that auto-expire.

**Characteristics:**

- TTL-based expiration (default: 5 minutes)
- Monotonic observation ordering
- Can be promoted to episodic memory via "attention"

```python
cog = CognitiveMemory("agent", "/tmp/cog_db")

# Record an observation
sid = cog.record_sensory(
    modality="error",           # Channel: "text", "code", "error", "log"
    raw_data="TypeError: expected str, got int",
    ttl_seconds=300,            # Expires in 5 minutes
)

# Get recent (non-expired) sensory items
items = cog.get_recent_sensory(limit=10)
# Returns list of SensoryItem ordered by observation_order DESC

# Promote to episodic memory (creates ATTENDED_TO edge)
ep_id = cog.attend_to_sensory(sid, reason="This error pattern keeps recurring")

# Clean up expired items
pruned = cog.prune_expired_sensory()
```

**Dataclass:** `SensoryItem`

| Field | Type | Description |
|-------|------|-------------|
| `node_id` | `str` | Unique graph identifier |
| `modality` | `str` | Observation channel |
| `raw_data` | `str` | Raw observation content |
| `observation_order` | `int` | Monotonic insertion order |
| `expires_at` | `float` | Unix timestamp for expiry |
| `created_at` | `datetime` | Creation time |

---

### 2. Working Memory

**Purpose:** Active task context with bounded capacity.

**Characteristics:**

- Fixed capacity of 20 slots per task
- Evicts lowest-relevance slot when full
- Scoped by `task_id`

```python
# Push a slot (auto-evicts if at capacity)
wid = cog.push_working(
    slot_type="goal",           # e.g., "goal", "constraint", "context"
    content="Deploy v2.0 to staging by Friday",
    task_id="sprint-42",
    relevance=0.9,              # Priority weight (higher = more relevant)
)

# Get all working memory for a task
slots = cog.get_working("sprint-42")
# Returns list of WorkingMemorySlot ordered by relevance DESC

# Clear working memory when task is done
cleared = cog.clear_working("sprint-42")
```

**Dataclass:** `WorkingMemorySlot`

| Field | Type | Description |
|-------|------|-------------|
| `node_id` | `str` | Unique graph identifier |
| `slot_type` | `str` | Categorisation of the slot |
| `content` | `str` | Payload |
| `relevance` | `float` | Priority weight (default 1.0) |
| `task_id` | `str` | Associated task |
| `created_at` | `datetime` | Creation time |

---

### 3. Episodic Memory

**Purpose:** Autobiographical event records that can be consolidated.

**Characteristics:**

- Temporal ordering via monotonic index
- Consolidation compresses old episodes into summaries
- Compressed episodes are excluded from default queries

```python
# Store an episode
eid = cog.store_episode(
    content="User requested API rate limiting feature",
    source_label="user-session",
    metadata={"sprint": 42},
)

# Retrieve episodes (excludes compressed by default)
episodes = cog.get_episodes(limit=20)

# Include compressed episodes
all_episodes = cog.get_episodes(limit=50, include_compressed=True)

# Consolidate oldest episodes into a summary
cons_id = cog.consolidate_episodes(
    batch_size=10,
    summarizer=lambda contents: "Summary: " + "; ".join(contents),
)
# Returns ConsolidatedEpisode node_id, or None if fewer than batch_size available
```

**Consolidation process:**

1. Fetch the `batch_size` oldest un-compressed episodes
2. Apply the `summarizer` function (or simple concatenation if None)
3. Create a `ConsolidatedEpisode` node with the summary
4. Mark original episodes as `compressed=True`
5. Create `CONSOLIDATES` edges from the consolidated node to each original

**Dataclass:** `EpisodicMemory`

| Field | Type | Description |
|-------|------|-------------|
| `node_id` | `str` | Unique graph identifier |
| `content` | `str` | Episode description |
| `source_label` | `str` | Origin label |
| `temporal_index` | `int` | Monotonic ordering |
| `compressed` | `bool` | Whether consolidated |
| `created_at` | `datetime` | Creation time |
| `metadata` | `dict` | Optional structured data |

---

### 4. Semantic Memory

**Purpose:** Distilled facts and knowledge, searchable by keyword.

**Characteristics:**

- Keyword-based search (CONTAINS on concept + content)
- Confidence scoring
- Tags and metadata
- Source provenance tracking

```python
# Store a fact
fid = cog.store_fact(
    concept="rate-limiting",
    content="API rate limit is 1000 requests per minute",
    confidence=0.95,
    source_id="ep_abc123",     # Optional provenance
    tags=["api", "config"],
    temporal_metadata={"source_date": "2024-01-15"},
)

# Search facts by keyword
facts = cog.search_facts("rate limit", limit=10, min_confidence=0.5)

# Get all facts
all_facts = cog.get_all_facts(limit=50)
```

**Dataclass:** `SemanticFact`

| Field | Type | Description |
|-------|------|-------------|
| `node_id` | `str` | Unique graph identifier |
| `concept` | `str` | Topic/concept |
| `content` | `str` | Factual content |
| `confidence` | `float` | Confidence (0.0-1.0) |
| `source_id` | `str` | Provenance reference |
| `tags` | `list[str]` | Categorisation tags |
| `metadata` | `dict` | Additional metadata |
| `created_at` | `datetime` | Creation time |

---

### 5. Procedural Memory

**Purpose:** Reusable step-by-step procedures with usage tracking.

**Characteristics:**

- Ordered step lists
- Prerequisites tracking
- Usage count auto-incremented on recall
- Keyword search on name and steps

```python
# Store a procedure
pid = cog.store_procedure(
    name="Deploy to production",
    steps=["Run test suite", "Build Docker image", "Push to registry", "Update k8s manifests"],
    prerequisites=["All tests passing", "Approval from tech lead"],
)

# Recall procedures (auto-increments usage_count)
procs = cog.recall_procedure("deploy production", limit=5)
for proc in procs:
    print(f"{proc.name}: {proc.steps} (used {proc.usage_count} times)")
```

**Dataclass:** `ProceduralMemory`

| Field | Type | Description |
|-------|------|-------------|
| `node_id` | `str` | Unique graph identifier |
| `name` | `str` | Procedure name |
| `steps` | `list[str]` | Ordered step list |
| `prerequisites` | `list[str]` | Required conditions |
| `usage_count` | `int` | Times recalled |
| `created_at` | `datetime` | Creation time |

---

### 6. Prospective Memory

**Purpose:** Future-oriented trigger-action pairs.

**Characteristics:**

- Three states: `pending` -> `triggered` -> `resolved`
- Priority-based ordering
- Keyword-overlap trigger matching

```python
# Store a trigger-action pair
pid = cog.store_prospective(
    description="Alert on deployment failure",
    trigger_condition="deployment failed error",
    action_on_trigger="Send alert to #ops-channel and rollback",
    priority=3,
)

# Check triggers against new content
triggered = cog.check_triggers("The deployment failed with timeout error")
for pm in triggered:
    print(f"TRIGGERED: {pm.description} -> {pm.action_on_trigger}")

# Mark as resolved after handling
cog.resolve_prospective(triggered[0].node_id)
```

**Trigger matching:** Simple keyword overlap -- if any word from `trigger_condition` appears in the content, the prospective memory is triggered.

**Dataclass:** `ProspectiveMemory`

| Field | Type | Description |
|-------|------|-------------|
| `node_id` | `str` | Unique graph identifier |
| `description` | `str` | Description |
| `trigger_condition` | `str` | Trigger text |
| `action_on_trigger` | `str` | Action to take |
| `status` | `str` | pending/triggered/resolved |
| `priority` | `int` | Priority level |
| `created_at` | `datetime` | Creation time |

---

## Statistics

```python
stats = cog.get_statistics()
# {
#     "sensory": 5,
#     "working": 12,
#     "episodic": 45,
#     "semantic": 120,
#     "procedural": 8,
#     "prospective": 3,
#     "total": 193,
# }
```

---

## Lifecycle

```python
# Create
cog = CognitiveMemory("my-agent", "/tmp/cog_db")

# Use...

# Close (releases Kuzu connection)
cog.close()
```

The `CognitiveMemory` instance manages a single Kuzu database connection. Call `close()` when done to release resources. Agent isolation is achieved via the `agent_id` column on every node table.
