# amplihack-memory-lib

**Standalone memory system for goal-seeking AI agents.**

amplihack-memory-lib provides a graph-based memory system built on [Kuzu](https://kuzudb.com/) that gives AI agents persistent, structured memory with cognitive science-inspired categories. It supports episodic, semantic, procedural, prospective, and working memory -- enabling agents to learn, recall, and reason across sessions.

---

## Quick Start

### Installation

```bash
# Using uv (recommended)
uv pip install -e .

# Using pip
pip install -e .
```

### Basic Usage

```python
from amplihack_memory import HierarchicalMemory

# Create a memory instance for your agent
mem = HierarchicalMemory("my-agent", "/tmp/my_agent_memory")

# Store knowledge
node_id = mem.store_knowledge(
    content="Python 3.12 introduced type parameter syntax",
    concept="python-features",
    confidence=0.95,
    tags=["python", "typing"],
)

# Retrieve related knowledge via Graph RAG
subgraph = mem.retrieve_subgraph("python type features")
print(subgraph.to_llm_context())
```

### Six-Type Cognitive Memory

```python
from amplihack_memory import CognitiveMemory

cog = CognitiveMemory("my-agent", "/tmp/cog_memory")

# Sensory: short-lived observations
cog.record_sensory("text", "User asked about deployment", ttl_seconds=300)

# Working: active task context (bounded capacity)
cog.push_working("goal", "Deploy v2.0 to staging", task_id="deploy-task")

# Episodic: autobiographical events
cog.store_episode("Deployment succeeded on staging", source_label="ci-run")

# Semantic: distilled facts
cog.store_fact("staging", "Staging environment uses port 8080", confidence=0.9)

# Procedural: reusable procedures
cog.store_procedure("deploy", ["Build image", "Push to registry", "Update k8s"])

# Prospective: future trigger-action pairs
cog.store_prospective(
    "Alert on failure",
    trigger_condition="deployment failed",
    action_on_trigger="Notify #ops channel",
)
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](architecture.md) | Package structure, core components, design philosophy, data flow |
| [Hierarchical Memory](hierarchical_memory.md) | Graph-based memory with 5 categories, SUPERSEDES/TRANSITIONED_TO edges, Graph RAG |
| [Cognitive Memory](cognitive_memory.md) | Six-type cognitive memory system (sensory, working, episodic, semantic, procedural, prospective) |
| [Kuzu Backend](kuzu_backend.md) | How the Kuzu graph database works, schema, query patterns |
| [API Reference](api_reference.md) | Complete API reference for all public classes |
| [Extending](extending.md) | Custom backends, new edge types, custom classifiers, integration patterns |

---

## Key Features

- **Graph-based knowledge storage** -- Facts stored as nodes, relationships as edges, enabling Graph RAG retrieval
- **Cognitive memory categories** -- Sensory, working, episodic, semantic, procedural, and prospective memory types
- **Auto-classification** -- Rule-based classifier assigns memory categories automatically
- **Temporal reasoning** -- SUPERSEDES and TRANSITIONED_TO edges track how knowledge evolves over time
- **Entity-centric retrieval** -- Extract entity names for targeted graph traversal
- **Contradiction detection** -- Automatically detect conflicting numerical values between facts
- **Similarity-based Graph RAG** -- Jaccard-based text similarity creates SIMILAR_TO edges for traversal
- **Dual backend** -- Kuzu graph database (default) with SQLite fallback
- **Security layer** -- Capability-based access control, credential scrubbing, query cost validation
- **Zero dependencies on amplihack** -- Fully standalone, requires only `kuzu` and Python stdlib

---

## Architecture at a Glance

```
amplihack_memory/
    __init__.py              # Public API exports
    hierarchical_memory.py   # HierarchicalMemory (Graph RAG)
    cognitive_memory.py      # CognitiveMemory (6-type cognitive)
    memory_types.py          # Dataclasses for cognitive memory types
    similarity.py            # Text similarity (Jaccard, reranking)
    entity_extraction.py     # Entity name extraction
    contradiction.py         # Contradiction detection
    connector.py             # MemoryConnector (backend factory)
    experience.py            # Experience data model
    store.py                 # ExperienceStore (high-level)
    security.py              # Security layer
    pattern_recognition.py   # Pattern detection
    semantic_search.py       # TF-IDF relevance scoring
    exceptions.py            # Custom exceptions
    backends/
        base.py              # Abstract MemoryBackend
        kuzu_backend.py      # Kuzu graph backend
        sqlite_backend.py    # SQLite fallback backend
```
