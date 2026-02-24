# Extending amplihack-memory-lib

This guide covers how to extend the library with custom backends, new edge types, custom classifiers, and integration patterns.

---

## Custom Memory Backends

### Implementing the MemoryBackend Interface

All backends must implement the abstract `MemoryBackend` class:

```python
from amplihack_memory.backends.base import MemoryBackend
from amplihack_memory.experience import Experience, ExperienceType


class MyCustomBackend(MemoryBackend):
    """Custom backend implementation."""

    def __init__(self, db_path, agent_name, max_memory_mb=100, enable_compression=True):
        self.db_path = db_path
        self.agent_name = agent_name
        self.max_memory_mb = max_memory_mb
        self.enable_compression = enable_compression
        self.initialize_schema()

    def initialize_schema(self):
        """Create storage schema."""
        # Set up your storage (tables, files, etc.)
        ...

    def store_experience(self, experience: Experience) -> str:
        """Store an experience. Returns experience_id."""
        ...

    def retrieve_experiences(self, limit=None, experience_type=None, min_confidence=0.0):
        """Retrieve experiences sorted by recency."""
        ...

    def search(self, query, experience_type=None, min_confidence=0.0, limit=10):
        """Search by text query."""
        ...

    def get_statistics(self) -> dict:
        """Return storage statistics."""
        ...

    def close(self):
        """Release resources."""
        ...

    def get_connection(self):
        """Return underlying connection object."""
        ...

    def cleanup(self, auto_compress=True, max_age_days=None, max_experiences=None):
        """Run cleanup operations."""
        ...
```

### Registering a Custom Backend

To use your backend with `MemoryConnector`, modify the factory logic:

```python
from amplihack_memory.connector import MemoryConnector


# Option 1: Direct backend injection
from amplihack_memory.backends.base import MemoryBackend

class MyConnector(MemoryConnector):
    def __init__(self, agent_name, backend="custom", **kwargs):
        if backend == "custom":
            self._backend = MyCustomBackend(
                db_path=kwargs.get("storage_path", "default_path"),
                agent_name=agent_name,
            )
        else:
            super().__init__(agent_name, backend=backend, **kwargs)
```

```python
# Option 2: Use the backend directly with ExperienceStore pattern
from amplihack_memory.store import ExperienceStore


class CustomExperienceStore:
    """ExperienceStore with custom backend."""

    def __init__(self, agent_name, backend_instance):
        self.agent_name = agent_name
        self._backend = backend_instance

    def add(self, experience):
        return self._backend.store_experience(experience)

    def search(self, query, **kwargs):
        return self._backend.search(query, **kwargs)
```

---

## New Edge Types

### Adding Edges to HierarchicalMemory

To add a new relationship type to the knowledge graph:

**Step 1:** Add the schema definition in `_init_schema()`:

```python
# In hierarchical_memory.py, inside _init_schema()
self.connection.execute("""
    CREATE REL TABLE IF NOT EXISTS CAUSED_BY(
        FROM SemanticMemory TO SemanticMemory,
        cause_type STRING,
        confidence DOUBLE
    )
""")
```

**Step 2:** Add a method to create the edge:

```python
def _create_causal_edge(self, effect_id: str, cause_id: str, cause_type: str, confidence: float):
    """Create a CAUSED_BY edge between two knowledge nodes."""
    try:
        self.connection.execute(
            """
            MATCH (effect:SemanticMemory), (cause:SemanticMemory)
            WHERE effect.memory_id = $eid AND cause.memory_id = $cid
            CREATE (effect)-[:CAUSED_BY {
                cause_type: $ct,
                confidence: $conf
            }]->(cause)
            """,
            {"eid": effect_id, "cid": cause_id, "ct": cause_type, "conf": confidence},
        )
    except Exception as e:
        logger.warning("Failed to create CAUSED_BY edge: %s", e)
```

**Step 3:** Include the edge in retrieval:

```python
# In retrieve_subgraph(), add traversal for the new edge type
# after the similarity traversal section:

causal_result = self.connection.execute(
    """
    MATCH (a:SemanticMemory)-[e:CAUSED_BY]-(b:SemanticMemory)
    WHERE a.memory_id IN $ids AND b.agent_id = $agent_id
    RETURN b.memory_id, b.concept, b.content, b.confidence, ...
    """,
    {"ids": list(seen_ids), "agent_id": self.agent_name},
)
```

### Adding Edges to CognitiveMemory

For CognitiveMemory, add the table to the `_REL_TABLES` list:

```python
# In cognitive_memory.py, add to _REL_TABLES list:
_REL_TABLES = [
    # ... existing tables ...
    """
    CREATE REL TABLE IF NOT EXISTS CAUSED_BY(
        FROM SemanticMemory TO SemanticMemory,
        cause_type STRING,
        confidence DOUBLE
    )
    """,
]
```

---

## Custom Classifiers

### Replacing the MemoryClassifier

The default `MemoryClassifier` uses keyword matching. To create a more sophisticated classifier:

```python
from amplihack_memory.hierarchical_memory import MemoryCategory


class LLMClassifier:
    """Memory classifier using an LLM for nuanced categorization."""

    def __init__(self, llm_client):
        self.llm = llm_client

    def classify(self, content: str, concept: str = "") -> MemoryCategory:
        """Classify using LLM."""
        prompt = f"""Classify this memory into one category:
        - episodic: Events that happened
        - semantic: Factual knowledge
        - procedural: Step-by-step procedures
        - prospective: Future plans or intentions
        - working: Active task context

        Content: {content}
        Concept: {concept}

        Category:"""

        result = self.llm.complete(prompt).strip().lower()

        try:
            return MemoryCategory(result)
        except ValueError:
            return MemoryCategory.SEMANTIC  # Safe default
```

To use it, inject it into the HierarchicalMemory instance:

```python
mem = HierarchicalMemory("my-agent", "/tmp/mem_db")
mem._classifier = LLMClassifier(my_llm_client)
```

### Custom Similarity Functions

Replace the default Jaccard similarity with embeddings:

```python
from amplihack_memory.similarity import compute_similarity


def embedding_similarity(node_a: dict, node_b: dict) -> float:
    """Compute similarity using vector embeddings."""
    vec_a = embed(node_a.get("content", ""))
    vec_b = embed(node_b.get("content", ""))
    return cosine_similarity(vec_a, vec_b)


# Monkey-patch or subclass HierarchicalMemory to use your function
# in _create_similarity_edges()
```

---

## Integration Patterns

### Using with LangChain

```python
from amplihack_memory import HierarchicalMemory


class AmplihackMemoryRetriever:
    """LangChain-compatible retriever backed by HierarchicalMemory."""

    def __init__(self, agent_name: str, db_path: str):
        self.mem = HierarchicalMemory(agent_name, db_path)

    def get_relevant_documents(self, query: str) -> list[dict]:
        """Retrieve relevant documents for a query."""
        subgraph = self.mem.retrieve_subgraph(query, max_nodes=10)
        return [
            {
                "page_content": node.content,
                "metadata": {
                    "concept": node.concept,
                    "confidence": node.confidence,
                    "source_id": node.source_id,
                    "tags": node.tags,
                },
            }
            for node in subgraph.nodes
        ]
```

### Using with Claude/Anthropic Agents

```python
from amplihack_memory import HierarchicalMemory


class AgentMemory:
    """Memory adapter for Claude-based agents."""

    def __init__(self, agent_name: str):
        self.mem = HierarchicalMemory(agent_name)

    def remember(self, content: str, concept: str = "", **kwargs):
        """Store knowledge from agent interactions."""
        return self.mem.store_knowledge(content, concept, **kwargs)

    def recall(self, query: str) -> str:
        """Retrieve knowledge as LLM context."""
        subgraph = self.mem.retrieve_subgraph(query)
        return subgraph.to_llm_context()

    def learn_from_episode(self, raw_content: str, source: str, facts: list[dict]):
        """Store an episode and derived facts with provenance."""
        ep_id = self.mem.store_episode(raw_content, source_label=source)
        for fact in facts:
            self.mem.store_knowledge(
                content=fact["content"],
                concept=fact.get("concept", ""),
                source_id=ep_id,
                tags=fact.get("tags", []),
            )
```

### Using with Multi-Agent Systems

```python
from amplihack_memory import HierarchicalMemory


# Each agent gets its own isolated memory
architect_mem = HierarchicalMemory("architect-agent")
builder_mem = HierarchicalMemory("builder-agent")
reviewer_mem = HierarchicalMemory("reviewer-agent")

# Share knowledge between agents via export/import
shared_knowledge = architect_mem.export_graph()
builder_mem.import_graph(shared_knowledge)
```

### Session-Scoped Memory with CognitiveMemory

```python
from amplihack_memory import CognitiveMemory


class SessionMemory:
    """Session-scoped memory using all six cognitive types."""

    def __init__(self, agent_name: str, session_id: str):
        self.cog = CognitiveMemory(agent_name, f"/tmp/session_{session_id}")
        self.session_id = session_id

    def observe(self, data: str, modality: str = "text"):
        """Record a sensory observation."""
        return self.cog.record_sensory(modality, data)

    def focus(self, goal: str, context: str):
        """Push items into working memory for the current task."""
        self.cog.push_working("goal", goal, task_id=self.session_id)
        self.cog.push_working("context", context, task_id=self.session_id)

    def note(self, event: str, source: str = ""):
        """Record an event."""
        return self.cog.store_episode(event, source_label=source or self.session_id)

    def learn(self, concept: str, fact: str, confidence: float = 0.9):
        """Store a learned fact."""
        return self.cog.store_fact(concept, fact, confidence=confidence)

    def remind_me(self, trigger: str, action: str):
        """Set a reminder for future conditions."""
        return self.cog.store_prospective(
            f"Reminder for {self.session_id}",
            trigger_condition=trigger,
            action_on_trigger=action,
        )

    def check_reminders(self, current_state: str):
        """Check if any reminders should fire."""
        return self.cog.check_triggers(current_state)

    def cleanup(self):
        """Prune expired sensory data and consolidate old episodes."""
        self.cog.prune_expired_sensory()
        self.cog.consolidate_episodes(batch_size=10)

    def close(self):
        self.cog.close()
```

---

## Best Practices

1. **One agent, one memory instance** -- Do not share HierarchicalMemory or CognitiveMemory instances across threads. Create separate instances if needed.

2. **Close when done** -- Always call `close()` or use context managers to release Kuzu connections.

3. **Use concepts consistently** -- Consistent concept labels improve similarity edge quality and retrieval accuracy.

4. **Tag strategically** -- Tags contribute to similarity scores (20% weight). Use consistent, meaningful tags.

5. **Provide source_id for provenance** -- Store episodes first, then derive facts with `source_id` pointing to the episode. This creates a traceable knowledge lineage.

6. **Use temporal_metadata for evolving knowledge** -- When facts change over time, provide `temporal_metadata` with `temporal_index` to enable automatic SUPERSEDES detection.
