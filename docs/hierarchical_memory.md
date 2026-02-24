# Hierarchical Memory

The `HierarchicalMemory` class is the primary memory system for AI agents that need structured, graph-based knowledge storage with relationship tracking and Graph RAG retrieval.

---

## Overview

HierarchicalMemory manages a Kuzu knowledge graph with two node types and four edge types:

**Node Types:**

- **SemanticMemory** -- Distilled facts (concept, content, confidence, tags, entity_name)
- **EpisodicMemory** -- Raw source content with provenance labels

**Edge Types:**

- **SIMILAR_TO** -- Text similarity between semantic nodes (auto-computed on store)
- **DERIVES_FROM** -- Provenance link from fact to source episode
- **SUPERSEDES** -- Temporal update (newer fact replaces older about same entity)
- **TRANSITIONED_TO** -- Explicit value transition chain for temporal reasoning

---

## Memory Categories

HierarchicalMemory uses five categories from the `MemoryCategory` enum:

| Category | Value | Description | Classification Keywords |
|----------|-------|-------------|------------------------|
| **EPISODIC** | `"episodic"` | Events that happened | happened, event, observed |
| **SEMANTIC** | `"semantic"` | Factual knowledge (default) | (anything not matching other categories) |
| **PROCEDURAL** | `"procedural"` | Step-by-step procedures | step, how to, procedure, recipe |
| **PROSPECTIVE** | `"prospective"` | Future plans/intentions | plan, goal, future, will, todo |
| **WORKING** | `"working"` | Active task context | (not auto-classified; set explicitly) |

The `MemoryClassifier` auto-assigns categories based on keyword matching. You can override by passing `category=` explicitly.

---

## Storing Knowledge

### Basic Storage

```python
from amplihack_memory import HierarchicalMemory, MemoryCategory

mem = HierarchicalMemory("my-agent", "/tmp/mem_db")

# Auto-classified as SEMANTIC (default)
nid = mem.store_knowledge(
    content="Python was created by Guido van Rossum",
    concept="python-history",
    confidence=0.95,
    tags=["python", "history"],
)

# Explicitly set category
nid = mem.store_knowledge(
    content="Step 1: Clone repo. Step 2: Run tests. Step 3: Deploy.",
    concept="deployment-procedure",
    category=MemoryCategory.PROCEDURAL,
)
```

### What Happens on Store

1. **Auto-classification** -- If no category given, `MemoryClassifier` assigns one based on keywords
2. **Entity extraction** -- `extract_entity_name()` finds proper nouns for entity-centric indexing
3. **Node creation** -- SemanticMemory node created in Kuzu with all fields
4. **Provenance** -- If `source_id` points to an existing EpisodicMemory node, a `DERIVES_FROM` edge is created
5. **Temporal updates** -- If `temporal_metadata` with `temporal_index > 0` is provided, the system checks for existing facts about the same entity and creates `SUPERSEDES` + `TRANSITIONED_TO` edges if a contradiction is detected
6. **Similarity edges** -- `compute_similarity()` runs against recent nodes; `SIMILAR_TO` edges are created for scores above 0.3

### Storing Episodes

Episodes are raw source content -- the provenance from which facts are derived:

```python
episode_id = mem.store_episode(
    content="In the 2024 Olympics, Norway won 15 gold medals...",
    source_label="news-article-2024",
)

# Later, store a fact derived from this episode
fact_id = mem.store_knowledge(
    content="Norway won 15 gold medals in 2024 Olympics",
    concept="norway-olympics",
    source_id=episode_id,  # Creates DERIVES_FROM edge
)
```

---

## SUPERSEDES Edges (Temporal Updates)

When knowledge evolves over time, SUPERSEDES edges track updates:

```python
# Day 1: Initial fact
mem.store_knowledge(
    content="Klaebo has 9 gold medals",
    concept="Klaebo medals",
    temporal_metadata={"temporal_index": 1, "source_date": "Day 1"},
)

# Day 2: Updated fact (auto-detected via contradiction)
mem.store_knowledge(
    content="Klaebo has 10 gold medals",
    concept="Klaebo medals",
    temporal_metadata={"temporal_index": 2, "source_date": "Day 2"},
)
# Result: (Day 2 fact)-[:SUPERSEDES]->(Day 1 fact)
```

**How supersession works:**

1. When a new fact has `temporal_index > 0`, the system searches for existing facts about the same entity
2. `detect_contradiction()` checks if the facts share concept words but have different numbers
3. If contradiction detected: the new fact SUPERSEDES the old one, and the old fact's confidence is reduced
4. A `TRANSITIONED_TO` edge is also created to enable multi-hop traversal of the value history

---

## TRANSITIONED_TO Edges (Value Chains)

TRANSITIONED_TO edges create explicit value transition chains:

```
(latest: "10 golds")-[:TRANSITIONED_TO]->(intermediate: "9 golds")
                                              |
                                    [:TRANSITIONED_TO]
                                              |
                                              v
                                    (first: "8 golds")
```

These edges store:

| Field | Description |
|-------|-------------|
| `from_value` | The newer value |
| `to_value` | The older value |
| `turn` | The temporal index of the newer fact |
| `transition_type` | Type of transition (e.g., `"update"`) |

The `to_llm_context(chronological=True)` method uses these edges to present a chronological history to the LLM.

---

## Similarity-Based Graph RAG

### How Similarity Works

On every `store_knowledge()` call, the system:

1. Fetches the 50 most recent SemanticMemory nodes for the same agent
2. Computes `compute_similarity()` between the new node and each existing node
3. Creates `SIMILAR_TO` edges for pairs with similarity > 0.3

The similarity score is a weighted composite:

```
score = 0.5 * word_similarity + 0.2 * tag_similarity + 0.3 * concept_similarity
```

Where each component uses Jaccard coefficients on tokenized text (lowercase, stop words removed).

### Retrieve Subgraph

```python
subgraph = mem.retrieve_subgraph(
    query="python type features",
    max_nodes=20,
    similarity_threshold=0.3,
)

# Get LLM-ready context
context = subgraph.to_llm_context()
print(context)
```

**Retrieval process:**

1. **Keyword matching** -- `CONTAINS` on concept and content fields (case-insensitive)
2. **Entity-centric retrieval** -- Extract entity name from query, match on `entity_name` field
3. **Merge and deduplicate** -- Combine results by `memory_id`
4. **Graph traversal** -- Follow SIMILAR_TO edges (1 hop) from matched nodes
5. **Temporal context** -- Follow SUPERSEDES and TRANSITIONED_TO chains
6. **Edge collection** -- Gather all edges between result nodes
7. **Return** -- `KnowledgeSubgraph(nodes, edges, query)`

### LLM Context Formatting

```python
# Default: sorted by confidence
context = subgraph.to_llm_context()

# Chronological: sorted by temporal_index
context = subgraph.to_llm_context(chronological=True)
```

The formatted output includes:

- Numbered facts with concept labels and confidence scores
- Source provenance markers `[Source: ...]`
- Chain position markers `[chain_position]`
- Transition history (from TRANSITIONED_TO edges)
- Contradiction warnings (from edges with contradiction metadata)
- Relationship summary (SIMILAR_TO, DERIVES_FROM, etc.)

---

## Entity-Centric Retrieval

Entity names are extracted from content and concept fields using regex heuristics:

```python
from amplihack_memory import extract_entity_name

name = extract_entity_name("Klaebo won 10 gold medals", "Klaebo medals")
# Returns: "klaebo"
```

During retrieval, the system:

1. Extracts entity name from the query
2. Matches against the `entity_name` field on SemanticMemory nodes
3. Merges with keyword-based results

This enables questions like "How many medals does Klaebo have?" to find all facts about Klaebo, even if the exact wording differs.

---

## Export / Import

```python
# Export the full graph to a dict
data = mem.export_graph()
# Returns: {"nodes": [...], "edges": [...], "agent_name": "...", "exported_at": "..."}

# Import into a different memory instance
other_mem = HierarchicalMemory("other-agent", "/tmp/other_db")
other_mem.import_graph(data)
```

---

## Protocol-Compatible Aliases

For interop with code that expects the `ExperienceStore` interface:

| Alias | Delegates To |
|-------|-------------|
| `store_fact(content, concept, confidence, **kwargs)` | `store_knowledge()` |
| `search_facts(query, max_nodes, **kwargs)` | `retrieve_subgraph()` -- returns list of dicts |
| `get_all_facts(limit)` | `get_all_knowledge()` -- returns list of dicts |

```python
# These are equivalent:
mem.store_knowledge("Python is great", "python")
mem.store_fact("Python is great", "python")

# search_facts returns dicts instead of KnowledgeSubgraph
facts = mem.search_facts("python")
# [{"content": "Python is great", "concept": "python", "confidence": 0.8, ...}]
```

---

## Static Utility Methods

```python
# Get recent discoveries from memory
discoveries = HierarchicalMemory.get_recent_discoveries(db_path, agent_name)

# Store a discovery
HierarchicalMemory.store_discovery(db_path, agent_name, discovery_dict)
```

These static methods provide backward-compatible access without requiring a full HierarchicalMemory instance.
