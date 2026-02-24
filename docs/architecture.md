# Architecture

## Overview

amplihack-memory-lib is organized into three layers, each building on the one below:

```
+------------------------------------------------------------+
|                     Public API Layer                        |
|  HierarchicalMemory  |  CognitiveMemory  |  ExperienceStore|
+------------------------------------------------------------+
|                   Shared Utilities Layer                    |
|  similarity  |  entity_extraction  |  contradiction        |
|  MemoryClassifier  |  pattern_recognition  |  security     |
+------------------------------------------------------------+
|                    Storage Backend Layer                    |
|  MemoryConnector  ->  KuzuBackend  |  SQLiteBackend        |
+------------------------------------------------------------+
|                     Kuzu / SQLite                           |
+------------------------------------------------------------+
```

---

## Core Components

### HierarchicalMemory (Graph RAG)

**File:** `hierarchical_memory.py`

The primary memory system for AI agents that need structured knowledge with relationships. It manages a Kuzu knowledge graph containing:

- **SemanticMemory nodes** -- Distilled facts with concept labels, confidence scores, tags, and entity names
- **EpisodicMemory nodes** -- Raw source content (episodes) with provenance labels
- **SIMILAR_TO edges** -- Computed via Jaccard text similarity at storage time
- **DERIVES_FROM edges** -- Link facts to their source episodes (provenance)
- **SUPERSEDES edges** -- Track temporal updates (newer fact replaces older)
- **TRANSITIONED_TO edges** -- Explicit value transition chains for temporal reasoning

Key methods:

| Method | Purpose |
|--------|---------|
| `store_knowledge()` | Store a fact node, auto-classify, compute similarity edges |
| `store_episode()` | Store a raw episode node |
| `retrieve_subgraph()` | Graph RAG retrieval: keyword match + similarity traversal |
| `get_all_knowledge()` | Get all fact nodes for an agent |
| `export_graph()` / `import_graph()` | Serialize/deserialize the full graph |

Protocol-compatible aliases (`store_fact`, `search_facts`, `get_all_facts`) provide interop with ExperienceStore consumers.

### CognitiveMemory (Six-Type)

**File:** `cognitive_memory.py`

A higher-level memory system modeled after human cognition with six distinct memory types, each stored in its own Kuzu node table:

| Memory Type | Table | Purpose | Lifecycle |
|-------------|-------|---------|-----------|
| **Sensory** | `SensoryMemory` | Raw short-lived observations | Auto-expires via TTL |
| **Working** | `WorkingMemory` | Active task context | Bounded capacity (20 slots), evicts lowest relevance |
| **Episodic** | `EpisodicMemory` | Autobiographical events | Consolidatable into summaries |
| **Semantic** | `SemanticMemory` | Distilled facts/knowledge | Persistent, searchable by keyword |
| **Procedural** | `ProceduralMemory` | Step-by-step procedures | Usage-count tracked, searchable |
| **Prospective** | `ProspectiveMemory` | Trigger-action pairs | Pending -> triggered -> resolved |

Relationship edges:

- **SIMILAR_TO** (SemanticMemory -> SemanticMemory)
- **DERIVES_FROM** (SemanticMemory -> EpisodicMemory)
- **PROCEDURE_DERIVES_FROM** (ProceduralMemory -> EpisodicMemory)
- **CONSOLIDATES** (ConsolidatedEpisode -> EpisodicMemory)
- **ATTENDED_TO** (SensoryMemory -> EpisodicMemory)

### ExperienceStore (Legacy/Simple)

**File:** `store.py`

A simpler, flat storage system for experience records. Uses `MemoryConnector` to delegate to either the Kuzu or SQLite backend. Provides:

- Automatic compression of old experiences
- Retention policies (age limit, count limit)
- Duplicate detection
- Full-text search
- Storage quota enforcement

### MemoryConnector (Backend Factory)

**File:** `connector.py`

Factory class that creates and manages the appropriate backend:

```python
# Kuzu backend (default)
connector = MemoryConnector(agent_name="my-agent", backend="kuzu")

# SQLite fallback
connector = MemoryConnector(agent_name="my-agent", backend="sqlite")
```

Each agent gets isolated storage under `~/.amplihack/memory/<agent_name>/`.

---

## Shared Utilities

### Similarity (`similarity.py`)

Deterministic text similarity without ML embeddings:

- **`compute_word_similarity(a, b)`** -- Jaccard coefficient on tokenized words minus stop words
- **`compute_tag_similarity(a, b)`** -- Jaccard coefficient on tag lists
- **`compute_similarity(node_a, node_b)`** -- Weighted composite: 0.5 * word + 0.2 * tag + 0.3 * concept
- **`rerank_facts_by_query(facts, query)`** -- Rerank retrieved facts by keyword relevance; boost temporal facts when query contains temporal cues

### Entity Extraction (`entity_extraction.py`)

Extracts proper nouns from text using regex heuristics:

- Handles apostrophe names (O'Brien), hyphenated names (Al-Hassan), multi-word names (Sarah Chen)
- Checks concept field first (more specific), then content
- Returns lowercase for consistent indexing

### Contradiction Detection (`contradiction.py`)

Detects when two facts about the same concept contain conflicting numerical values:

- Requires overlapping concept words (at least one meaningful word in common)
- Extracts numbers from both facts
- Flags when facts have numbers unique to each (potential update/conflict)

### MemoryClassifier (`hierarchical_memory.py`)

Rule-based keyword classifier:

| Keywords | Category |
|----------|----------|
| step, how to, procedure, recipe | PROCEDURAL |
| plan, goal, future, will, todo | PROSPECTIVE |
| happened, event, observed | EPISODIC |
| (default) | SEMANTIC |

### Security Layer (`security.py`)

- **AgentCapabilities** -- Capability-based access control (scope levels, allowed types, query cost limits)
- **CredentialScrubber** -- Regex-based detection and redaction of API keys, passwords, tokens, SSH keys, DB URLs
- **QueryValidator** -- SQL query cost estimation and safety validation
- **SecureMemoryBackend** -- Wrapper that enforces all security policies

### Pattern Recognition (`pattern_recognition.py`)

- **PatternDetector** -- Tracks recurring patterns across discoveries, recognizes when threshold is reached
- **`recognize_patterns()`** -- Batch pattern recognition with known-pattern filtering
- Confidence formula: `min(0.5 + occurrences * 0.1, 0.95)`, adjusted by validation success rate

---

## Design Philosophy

### Ruthless Simplicity

Every component has a single, clear purpose. No unnecessary abstractions. The similarity module uses Jaccard coefficients instead of ML embeddings -- simple, deterministic, and sufficient for the use case.

### Zero-BS Implementation

No stubs, no placeholders, no fake implementations. Every function works or does not exist. The security layer actually scrubs credentials. The pattern detector actually tracks occurrences.

### Regeneratable (Bricks & Studs)

Each module is a self-contained "brick" with a well-defined public API ("stud"). The similarity module, entity extraction, and contradiction detection are all independent -- they can be replaced, tested, or regenerated from their specification without affecting other modules.

---

## Data Flow

### Store Knowledge Flow

```
User calls store_knowledge(content, concept, ...)
    |
    v
MemoryClassifier assigns category (if not given)
    |
    v
extract_entity_name() extracts proper nouns
    |
    v
CREATE SemanticMemory node in Kuzu
    |
    +---> If source_id: CREATE DERIVES_FROM edge
    |
    +---> If temporal: _detect_supersedes()
    |       |
    |       +---> Find existing facts for same entity
    |       +---> detect_contradiction() checks for conflicts
    |       +---> CREATE SUPERSEDES edge + TRANSITIONED_TO edge
    |
    +---> _create_similarity_edges()
            |
            +---> compute_similarity() against recent nodes
            +---> CREATE SIMILAR_TO edges for scores > 0.3
```

### Retrieve Subgraph Flow (Graph RAG)

```
User calls retrieve_subgraph(query, max_nodes=20)
    |
    v
Keyword matching: CONTAINS on concept + content
    |
    v
Entity-centric retrieval: extract_entity_name(query)
    +---> Match on entity_name field
    |
    v
Merge direct matches (deduped by memory_id)
    |
    v
Graph traversal: follow SIMILAR_TO edges (1 hop)
    |
    v
Follow SUPERSEDES chain for temporal context
    |
    v
Follow TRANSITIONED_TO chain for value transitions
    |
    v
Collect all edges between result nodes
    |
    v
Return KnowledgeSubgraph(nodes, edges, query)
    |
    v
User calls subgraph.to_llm_context() for LLM-ready text
```
