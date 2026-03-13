# amplihack-memory

Standalone memory system for goal-seeking agents. Provides persistent, queryable
memory with cognitive modeling, hierarchical organization, and graph-based
knowledge storage.

[![Crates.io](https://img.shields.io/crates/v/amplihack-memory)](https://crates.io/crates/amplihack-memory)
[![docs.rs](https://docs.rs/amplihack-memory/badge.svg)](https://docs.rs/amplihack-memory)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Architecture Overview

```text
┌─────────────────────────────────────────────────────────────────┐
│                        Application Layer                        │
├────────────────┬──────────────────┬──────────────────────────────┤
│ ExperienceStore│ CognitiveMemory  │ HierarchicalMemory           │
│  (high-level)  │  (6 memory types)│  (knowledge graph / GraphRAG)│
├────────────────┴──────────────────┴──────────────────────────────┤
│                      MemoryConnector                            │
│          Factory — backend lifecycle & routing                   │
├──────────────────────────┬──────────────────────────────────────┤
│      Storage Backends    │          Graph Layer                  │
│  ┌──────────────────┐    │  ┌─────────────────────────────┐     │
│  │  SqliteBackend   │    │  │  InMemoryGraphStore          │     │
│  │  (FTS5 search)   │    │  │  KuzuGraphStore (feat: kuzu) │     │
│  ├──────────────────┤    │  │  HiveGraphStore               │    │
│  │  KuzuBackend     │    │  │  FederatedGraphStore          │    │
│  │  (feat: kuzu)    │    │  └─────────────────────────────┘     │
│  ├──────────────────┤    │                                      │
│  │  LadybugBackend  │    │                                      │
│  │  (feat: ladybug) │    │                                      │
│  └──────────────────┘    │                                      │
├──────────────────────────┴──────────────────────────────────────┤
│  Cross-cutting: Security │ Semantic Search │ Pattern Recognition │
│  CredentialScrubber · AgentCapabilities · TF-IDF · PatternDetector│
└─────────────────────────────────────────────────────────────────┘
```

Data flows from application code through **MemoryConnector** (or the higher-level
**ExperienceStore**) into a pluggable storage backend. The **Graph Layer** provides
associative recall across experiences, while **CognitiveMemory** organises data
into six biologically-inspired memory types. **Security**, **Semantic Search**, and
**Pattern Recognition** cut across all layers.

## Installation

Add the crate with default features (SQLite backend):

```sh
cargo add amplihack-memory
```

Enable optional backends:

```sh
# Kuzu graph database backend
cargo add amplihack-memory --features kuzu

# Native LadybugDB graph backend
cargo add amplihack-memory --features ladybug

# Python extension module (used by maturin)
cargo add amplihack-memory --features python
```

**Minimum Supported Rust Version (MSRV):** 1.80

## Feature Flags

| Feature    | Default | What it enables                                                  |
|------------|---------|------------------------------------------------------------------|
| *(none)*   | ✅      | SQLite backend with FTS5 full-text search, `InMemoryGraphStore`  |
| `kuzu`     |         | Kuzu graph database via PyO3 bridge — `KuzuBackend` + `KuzuGraphStore` |
| `ladybug`  |         | Native LadybugDB graph backend — `LadybugBackend`               |
| `python`   |         | PyO3 extension module exposing the full API to Python            |

> **Note:** `kuzu` and `ladybug` are mutually exclusive — enabling both produces a
> compile error.

## Backend Comparison

| Backend       | Persistence | Performance          | Graph Support          | Best For                        |
|---------------|-------------|----------------------|------------------------|---------------------------------|
| **SQLite**    | Disk        | Fast (FTS5 indexing) | Via separate graph layer | Default / single-agent workloads |
| **Kuzu**      | Disk        | Optimised for graphs | Native graph queries   | Complex relationship traversals |
| **LadybugDB** | Disk       | Native Rust graph    | Built-in               | Rust-native graph pipelines     |

All backends implement the `MemoryBackend` trait, so switching requires only a
`BackendType` change — no application-code rewrites.

## Quick Start

### Creating a connector and storing experiences

```rust
use amplihack_memory::{MemoryConnector, Experience, ExperienceType};
use std::path::Path;

fn main() -> amplihack_memory::Result<()> {
    // Create a connector with SQLite backend (default)
    let mut connector = MemoryConnector::new(
        "my-agent",
        Some(Path::new("/tmp/agent-memory")),
        256, // max storage in MB
    )?;

    // Store a successful experience
    let exp = Experience::new(
        ExperienceType::Success,
        "Parsed user query",        // context (max 500 chars)
        "Extracted 3 intent slots",  // outcome (max 1000 chars)
        0.92,                        // confidence ∈ [0.0, 1.0]
    );
    let id = connector.store_experience(&exp)?;
    println!("Stored: {id}");

    Ok(())
}
```

### Searching experiences

```rust
// Full-text search with optional filters
let results = connector.search(
    "intent slots",          // query
    Some(ExperienceType::Success), // filter by type (or None)
    Some(0.5),               // min confidence
    10,                      // max results
)?;

for exp in &results {
    println!("[{:.0}%] {} → {}",
        exp.confidence * 100.0, exp.context, exp.outcome);
}
```

### Semantic search with relevance scoring

```rust
use amplihack_memory::retrieve_relevant_experiences;

let relevant = retrieve_relevant_experiences(
    &all_experiences,
    "How do I parse queries?",
    5,    // top-k
    0.1,  // min similarity
);
// Results ranked by Jaccard similarity × type weight × recency decay
```

### Using cognitive memory (six types)

```rust
use amplihack_memory::{
    CognitiveMemory, SensoryItem, WorkingMemorySlot,
    EpisodicMemory, SemanticFact, ProceduralMemory, ProspectiveMemory,
};

let mut memory = CognitiveMemory::new();

// Sensory — raw short-lived observations
memory.add_sensory(SensoryItem {
    modality: "visual".into(),
    raw_data: "User clicked submit button".into(),
    observation_order: 1,
    expires_at: chrono::Utc::now() + chrono::Duration::seconds(30),
});

// Working — bounded active-task context
memory.add_working(WorkingMemorySlot {
    slot_type: "task-context".into(),
    content: "Processing checkout flow".into(),
    relevance: 0.95,
    task_id: "task-42".into(),
});

// Episodic — autobiographical event records
memory.add_episodic(EpisodicMemory {
    content: "Resolved cart conflict by merging".into(),
    source_label: "checkout-agent".into(),
    temporal_index: 1,
    compressed: false,
});

// Semantic — long-lived knowledge facts
memory.add_semantic(SemanticFact {
    concept: "cart-merge".into(),
    content: "When items conflict, prefer the newer addition".into(),
    confidence: 0.88,
    source_id: "exp_20250101_120000_abc123".into(),
    tags: vec!["checkout".into(), "conflict-resolution".into()],
});

// Procedural — reusable step-by-step procedures
memory.add_procedural(ProceduralMemory {
    name: "resolve-cart-conflict".into(),
    steps: vec!["Compare timestamps".into(), "Keep newer item".into()],
    prerequisites: vec!["cart-access".into()],
    usage_count: 0,
});

// Prospective — future trigger → action pairs
memory.add_prospective(ProspectiveMemory {
    description: "Alert on cart abandonment".into(),
    trigger_condition: "no_activity > 15min".into(),
    action_on_trigger: "Send reminder email".into(),
    status: "active".into(),
    priority: 1,
});
```

### Pattern detection

```rust
use amplihack_memory::{PatternDetector, Experience, ExperienceType};

let mut detector = PatternDetector::new(
    3,    // occurrences before a pattern is recognised
    0.6,  // minimum confidence threshold
);

// Feed discoveries into the detector
for exp in &experiences {
    detector.add_discovery(exp);
}

// Retrieve recognised patterns (returned as Experience objects)
let patterns = detector.get_recognized_patterns(0.6);
for p in &patterns {
    println!("Pattern: {} (confidence {:.0}%)", p.context, p.confidence * 100.0);
}
```

### Security — capability-based access control

```rust
use amplihack_memory::{
    SecureMemoryBackend, AgentCapabilities, ScopeLevel, CredentialScrubber,
};

// Define what this agent is allowed to do
let caps = AgentCapabilities {
    scope: ScopeLevel::CrossSessionRead,
    allowed_experience_types: vec![ExperienceType::Success, ExperienceType::Insight],
    max_query_cost: 100,
    can_access_patterns: true,
    memory_quota_mb: 64,
};

let secure = SecureMemoryBackend::new(backend, caps);

// Credential scrubbing — automatically redacts secrets
let scrubber = CredentialScrubber::default();
let (clean_text, had_secrets) = scrubber.scrub_text(
    "Connect with AKIA1234567890ABCDEF and sk-proj-abc123"
);
// clean_text: "Connect with [REDACTED-AWS-KEY] and [REDACTED-OPENAI-KEY]"
```

## Python Usage

Build with [maturin](https://www.maturin.rs/) (requires the `python` feature):

```sh
maturin develop --features python
```

### Basic usage from Python

```python
from amplihack_memory_rs import PyMemoryConnector, PyExperience

# Create a connector
conn = PyMemoryConnector("my-agent", "/tmp/agent-memory", 256)

# Store an experience
exp = PyExperience("success", "Parsed query", "Found 3 slots", 0.92)
exp_id = conn.store_experience(exp)
print(f"Stored: {exp_id}")

# Search
results = conn.search("query parsing", None, 0.5, 10)
for r in results:
    print(f"  [{r.confidence:.0%}] {r.context}")
```

### Cognitive memory and pattern detection

```python
from amplihack_memory_rs import PyCognitiveMemory, PyPatternDetector

memory = PyCognitiveMemory()
# Add items to each of the six memory types via add_sensory(),
# add_working(), add_episodic(), add_semantic(), add_procedural(),
# add_prospective() methods.

detector = PyPatternDetector(threshold=3, min_confidence=0.6)
# Feed experiences, then query recognised patterns.
```

### Utility functions

```python
from amplihack_memory_rs import (
    scrub_credentials,
    jaccard_similarity,
    detect_contradiction,
    extract_entities,
)

clean, had_secrets = scrub_credentials("key=AKIA1234567890ABCDEF")
sim = jaccard_similarity("hello world", "hello there")
contradiction = detect_contradiction("sky is blue", "sky is green", "sky", "sky")
```

## Module Reference

| Module                 | Description                                                      |
|------------------------|------------------------------------------------------------------|
| `backends`             | `MemoryBackend` / `ExperienceBackend` traits and implementations (SQLite, Kuzu, LadybugDB) |
| `cognitive_memory`     | Six-type cognitive memory system backed by an in-memory graph    |
| `connector`            | `MemoryConnector` — factory for backend lifecycle management     |
| `contradiction`        | Contradiction detection between semantic facts                   |
| `entity_extraction`    | Entity-name extraction from free text                            |
| `experience`           | `Experience` and `ExperienceType` data model                     |
| `graph`                | `GraphStore` trait with `InMemoryGraphStore`, `HiveGraphStore`, `FederatedGraphStore`, `KuzuGraphStore` |
| `hierarchical_memory`  | `HierarchicalMemory` — knowledge graph for Graph RAG retrieval   |
| `memory_types`         | `MemoryCategory` enum and structs for the six cognitive types    |
| `pattern_recognition`  | `PatternDetector` and pattern-confidence scoring                 |
| `python`               | PyO3 bindings (behind `python` feature flag)                     |
| `security`             | `SecureMemoryBackend`, `AgentCapabilities`, `CredentialScrubber`, `QueryValidator` |
| `semantic_search`      | `SemanticSearchEngine`, TF-IDF / Jaccard relevance scoring       |
| `similarity`           | Deterministic text-similarity functions (Jaccard word/tag)       |
| `store`                | `ExperienceStore` — high-level wrapper with quota & retention    |
| `errors`               | `MemoryError` enum and `Result` type alias                       |

## Contributing

1. **Pre-commit hooks** — run `cargo fmt --check && cargo clippy -- -D warnings`
   before committing.
2. **Tests** — `cargo test` must pass. Integration tests live in `tests/`.
3. **Benchmarks** — `cargo bench` (uses Criterion). Benchmark sources are in `benches/`.
4. **Feature matrix** — test with `--features kuzu`, `--features ladybug`, and
   default features separately.

See [`AGENTS.md`](../../AGENTS.md) in the repository root for full contributor
guidelines.

## License

MIT
