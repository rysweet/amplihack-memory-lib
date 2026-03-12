# amplihack-memory-lib - Quick Reference Guide

## What is This?
A **graph-based memory system for AI agents** using Kuzu (default) or SQLite. Provides persistent, structured memory with temporal reasoning, entity extraction, and Graph RAG - all without ML embeddings.

## Quick Start

### 1. Graph RAG Memory (Primary)
```python
from amplihack_memory import HierarchicalMemory

mem = HierarchicalMemory("my-agent", "/tmp/memory")

# Store knowledge (auto-classified, similarity edges computed)
mem.store_knowledge(
    content="Python 3.12 has type parameters",
    concept="python-features",
    confidence=0.95,
    tags=["python", "typing"]
)

# Retrieve via Graph RAG (keyword + similarity + temporal context)
subgraph = mem.retrieve_subgraph("python type features", max_nodes=20)
print(subgraph.to_llm_context())
```

### 2. Cognitive Memory (6-Type)
```python
from amplihack_memory import CognitiveMemory

cog = CognitiveMemory("agent", "/tmp/memory")

cog.record_sensory("error", "TypeError", ttl_seconds=300)  # Auto-expires
cog.push_working("goal", "Fix error", task_id="task1")     # Bounded capacity
cog.store_episode("Fixed by adding str()")                 # Autobiographical
cog.store_fact("python", "str() converts values")          # Persistent
cog.store_procedure("fix-type", ["Check", "Convert", "Test"])  # Steps
cog.store_prospective("Type alert", "TypeError", "Validate input")  # Reminders
```

### 3. Simple Store
```python
from amplihack_memory import ExperienceStore, Experience, ExperienceType

store = ExperienceStore("agent")

exp = Experience(
    experience_type=ExperienceType.SUCCESS,
    context="Task completed",
    outcome="All tests passed",
    confidence=0.95
)
store.add(exp)

results = store.search("tests passed", limit=10)
stats = store.get_statistics()
```

## Architecture

```
HierarchicalMemory / CognitiveMemory / ExperienceStore
           ↓
Shared Utilities (similarity, entity extraction, patterns, security)
           ↓
MemoryConnector (backend factory)
           ↓
KuzuBackend (default) or SQLiteBackend
           ↓
Kuzu Graph Database or SQLite
```

## Core Classes & APIs

### HierarchicalMemory
- `store_knowledge(content, concept, confidence, tags, source_id)` → memory_id
- `retrieve_subgraph(query, max_nodes=20)` → KnowledgeSubgraph
- `retrieve_by_entity(entity_name, limit)` → list[KnowledgeNode]
- `search_by_concept(concept, limit)` → list[KnowledgeNode]
- `export_to_json()` / `import_from_json(data, merge)`
- `get_statistics()` → dict (nodes, edges, by_category, by_entity)

### CognitiveMemory
- `record_sensory(modality, raw_data, ttl_seconds)` → id
- `push_working(slot_type, content, task_id, relevance)` → id
- `store_episode(content, source_label, metadata)` → id
- `store_fact(concept, content, confidence, source_id, tags)` → id
- `store_procedure(name, steps, prerequisites)` → id
- `store_prospective(description, trigger, action, priority)` → id
- `search_facts(query, use_vector, min_confidence, limit)` → list
- `check_triggers(content)` → list (matching prospectives)

### ExperienceStore
- `add(experience)` → id
- `search(query, experience_type, min_confidence, limit)` → list
- `get_statistics()` → dict

### Utilities
- `compute_similarity(node_a, node_b)` → float (0-1)
- `extract_entity_name(content, concept)` → str
- `detect_contradiction(content_a, content_b, concept_a, concept_b)` → dict
- `rerank_facts_by_query(facts, query, top_k)` → reranked list

## Data Models

### Experience
```python
Experience(
    experience_type: ExperienceType,  # SUCCESS|FAILURE|PATTERN|INSIGHT
    context: str,                      # Max 500 chars
    outcome: str,                      # Max 1000 chars
    confidence: float,                 # 0.0-1.0
    timestamp: datetime = None,
    experience_id: str = None,         # Auto-generated
    metadata: dict = None,
    tags: list[str] = None
)
```

### KnowledgeNode
```python
KnowledgeNode(
    memory_id: str,
    category: MemoryCategory,          # EPISODIC|SEMANTIC|PROCEDURAL|PROSPECTIVE|SENSORY|WORKING
    entity_name: str,
    content: str,
    concept: str,
    confidence: float,
    tags: list[str],
    created_at: datetime
)
```

### Memory Categories (Auto-Classified)
- **EPISODIC**: happened, event, observed
- **PROCEDURAL**: step, how to, procedure, recipe
- **PROSPECTIVE**: plan, goal, future, will, todo
- **SEMANTIC**: default (distilled facts)
- **SENSORY**: short-lived observations
- **WORKING**: active task context

## Edge Types
- `SIMILAR_TO`: Computed via Jaccard similarity (score > 0.3)
- `DERIVES_FROM`: Facts linked to source episodes
- `SUPERSEDES`: Newer fact replaces older (temporal)
- `TRANSITIONED_TO`: Value transition chains

## Security

```python
from amplihack_memory import (
    SecureMemoryBackend, AgentCapabilities,
    ScopeLevel, CredentialScrubber, QueryValidator
)

# Define capabilities
caps = AgentCapabilities(
    scope=ScopeLevel.SESSION_ONLY,
    allowed_experience_types=[ExperienceType.SUCCESS],
    max_query_cost=50,
    can_access_patterns=False,
    memory_quota_mb=10
)

# Wrap store
secure = SecureMemoryBackend(store, caps)

# Auto-scrubs credentials
scrubber = CredentialScrubber()
scrubbed_exp, modified = scrubber.scrub_experience(exp)

# Validate queries
validator = QueryValidator()
validator.validate_query(sql, max_cost=100)
```

## Backends

### Kuzu (Default)
- Graph-native, relationship tracking
- Cypher queries, BFS traversal
- Best for: temporal reasoning, multi-hop queries
- Init: ~0.13s

### SQLite (Fallback)
- Single-file storage, mature
- FTS5 full-text search
- Best for: simple queries, backward compatibility
- Init: ~0.05s

**Switch backend:**
```python
store = ExperienceStore("agent", backend="sqlite")  # or "kuzu"
```

## Storage & Defaults

- **Location**: `~/.amplihack/memory/<agent_name>/`
- **Max size**: 100MB (configurable)
- **Auto-compression**: Enabled by default
- **Retention**: No limit by default (configurable)

## Similarity Algorithm

**Jaccard coefficient on tokenized words (minus stop words):**
- Word similarity: intersection / union of tokens
- Tag similarity: intersection / union of tags
- Concept similarity: word similarity on concept field
- **Composite**: 0.5×word + 0.2×tag + 0.3×concept

No ML embeddings needed - deterministic and fast!

## Test Coverage

- **180/187 tests pass** (96.3%)
- **195,000 lines of test code** (20:1 test-to-code ratio)
- All core features tested: memory types, backends, security, utilities

## Public API Exports

```python
# Memory systems
from amplihack_memory import (
    HierarchicalMemory, CognitiveMemory, ExperienceStore,
    KnowledgeNode, KnowledgeEdge, KnowledgeSubgraph, MemoryClassifier
)

# Data models
from amplihack_memory import (
    Experience, ExperienceType,
    MemoryCategory, SensoryItem, WorkingMemorySlot,
    EpisodicMemory, SemanticFact, ProceduralMemory,
    ProspectiveMemory, ConsolidatedEpisode
)

# Utilities
from amplihack_memory import (
    compute_similarity, compute_word_similarity, compute_tag_similarity,
    rerank_facts_by_query, extract_entity_name, detect_contradiction
)

# Storage
from amplihack_memory import MemoryConnector, ExperienceStore

# Security
from amplihack_memory import (
    AgentCapabilities, ScopeLevel, CredentialScrubber,
    QueryValidator, SecureMemoryBackend,
    SecurityViolationError, QueryCostExceededError
)

# Exceptions
from amplihack_memory import (
    MemoryError, ExperienceNotFoundError, InvalidExperienceError,
    MemoryQuotaExceededError
)
```

## Dependencies

- **Required**: `kuzu >= 0.11.0`
- **Optional**: `sentence-transformers >= 2.2.0` (for vector embeddings)
- **Python**: 3.10+
- **External**: None beyond kuzu (fully standalone)

## Examples

```bash
# Basic usage
python examples/basic_usage.py

# Secure memory with capabilities
python examples/secure_memory_usage.py
```

## Version

**0.4.0** - Dual-backend (Kuzu default + SQLite fallback), production-ready

## Philosophy

- **Ruthlessly Simple**: Clear APIs, minimal abstractions
- **Zero-BS**: Every function works end-to-end
- **Regeneratable**: Self-contained modules, well-defined boundaries
