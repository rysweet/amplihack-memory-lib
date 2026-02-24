# amplihack-memory-lib

[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://rysweet.github.io/amplihack-memory-lib/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Standalone memory system for goal-seeking AI agents.**

A graph-based memory library built on [Kuzu](https://kuzudb.com/) that gives AI agents persistent, structured memory with cognitive science-inspired categories. Store facts, track how knowledge evolves over time, and retrieve context via Graph RAG -- all without ML embeddings.

**[Full Documentation](https://rysweet.github.io/amplihack-memory-lib/)**

## Features

- **Graph-based knowledge storage** -- Facts as nodes, relationships as edges, enabling Graph RAG retrieval
- **Six cognitive memory types** -- Sensory, working, episodic, semantic, procedural, and prospective
- **Auto-classification** -- Rule-based classifier assigns memory categories automatically
- **Temporal reasoning** -- SUPERSEDES and TRANSITIONED_TO edges track how knowledge evolves
- **Entity-centric retrieval** -- Entity name extraction for targeted graph traversal
- **Contradiction detection** -- Automatic detection of conflicting numerical values
- **Similarity-based Graph RAG** -- Jaccard text similarity creates SIMILAR_TO edges
- **Dual backend** -- Kuzu graph database (default) with SQLite fallback
- **Security layer** -- Capability-based access control, credential scrubbing, query validation
- **Zero external dependencies** -- Only `kuzu` and Python stdlib (no amplihack dependency)

## Installation

```bash
# Using uv (recommended)
uv pip install -e .

# Using pip
pip install -e .
```

## Quick Start

### HierarchicalMemory (Graph RAG)

```python
from amplihack_memory import HierarchicalMemory

mem = HierarchicalMemory("my-agent", "/tmp/my_agent_memory")

# Store knowledge (auto-classified, similarity edges computed)
mem.store_knowledge(
    content="Python 3.12 introduced type parameter syntax",
    concept="python-features",
    confidence=0.95,
    tags=["python", "typing"],
)

# Retrieve via Graph RAG (keyword + similarity traversal)
subgraph = mem.retrieve_subgraph("python type features")
print(subgraph.to_llm_context())
```

### CognitiveMemory (Six-Type)

```python
from amplihack_memory import CognitiveMemory

cog = CognitiveMemory("my-agent", "/tmp/cog_memory")

# Sensory (auto-expires)
cog.record_sensory("error", "TypeError: expected str", ttl_seconds=300)

# Working (bounded capacity per task)
cog.push_working("goal", "Fix the type error", task_id="debug-task")

# Episodic (consolidatable)
cog.store_episode("Fixed TypeError by adding str() cast", source_label="debug-session")

# Semantic (persistent facts)
cog.store_fact("python-types", "str() converts any value to string", confidence=0.9)

# Procedural (usage-tracked)
cog.store_procedure("fix-type-error", ["Check types", "Add conversion", "Test"])

# Prospective (trigger-action pairs)
cog.store_prospective("Type alert", "TypeError", "Check input validation")
```

### ExperienceStore (Simple)

```python
from amplihack_memory import ExperienceStore, Experience, ExperienceType

store = ExperienceStore("my-agent")

exp = Experience(
    experience_type=ExperienceType.SUCCESS,
    context="Analyzed codebase structure",
    outcome="Found 47 Python files",
    confidence=0.95,
)
store.add(exp)

results = store.search("codebase analysis")
```

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](https://rysweet.github.io/amplihack-memory-lib/architecture/) | Package structure, components, design philosophy, data flow |
| [Hierarchical Memory](https://rysweet.github.io/amplihack-memory-lib/hierarchical_memory/) | Graph RAG, SUPERSEDES edges, temporal reasoning |
| [Cognitive Memory](https://rysweet.github.io/amplihack-memory-lib/cognitive_memory/) | Six cognitive memory types with full lifecycle |
| [Kuzu Backend](https://rysweet.github.io/amplihack-memory-lib/kuzu_backend/) | Schema, query patterns, performance characteristics |
| [API Reference](https://rysweet.github.io/amplihack-memory-lib/api_reference/) | Complete API for all public classes |
| [Extending](https://rysweet.github.io/amplihack-memory-lib/extending/) | Custom backends, edge types, classifiers, integrations |

## Architecture

```
amplihack_memory/
    hierarchical_memory.py   # HierarchicalMemory (Graph RAG)
    cognitive_memory.py      # CognitiveMemory (6-type cognitive)
    similarity.py            # Jaccard similarity, reranking
    entity_extraction.py     # Entity name extraction
    contradiction.py         # Contradiction detection
    connector.py             # MemoryConnector (backend factory)
    store.py                 # ExperienceStore (high-level)
    security.py              # Access control, credential scrubbing
    backends/
        kuzu_backend.py      # Kuzu graph backend
        sqlite_backend.py    # SQLite fallback
```

## No Amplihack Dependencies

This library is completely standalone. It requires only:

- `kuzu>=0.3.0` for graph database operations
- Python 3.10+ standard library

## Philosophy

- **Ruthlessly Simple**: Minimal API surface, clear contracts, no ML embeddings needed
- **Zero-BS Implementation**: No stubs, no placeholders -- every function works
- **Regeneratable**: Self-contained modules that can be rebuilt from specification
