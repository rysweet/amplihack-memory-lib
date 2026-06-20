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
│  │  (feat: kuzu)    │    │  │  LbugGraphStore (persistent) │     │
│  ├──────────────────┤    │  └─────────────────────────────┘     │
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

**Minimum Supported Rust Version (MSRV):** 1.85

## Feature Flags

| Feature    | Default | What it enables                                                  |
|------------|---------|------------------------------------------------------------------|
| *(none)*   | ✅      | SQLite backend with FTS5 full-text search, `InMemoryGraphStore`  |
| `kuzu`     |         | Kuzu graph database via PyO3 bridge — `KuzuBackend` + `KuzuGraphStore` |
| `ladybug`  |         | Native LadybugDB graph backend — `LadybugBackend`               |
| `persistent` |       | Durable `CognitiveMemory` over the published `lbug` crate — `LbugGraphStore` |
| `python`   |         | PyO3 extension module exposing the full API to Python            |

> **Note:** `kuzu` and `ladybug` are mutually exclusive — enabling both produces a
> compile error.
>
> **Note:** `persistent` compiles the LadybugDB (Kùzu) C++ engine from source via
> the published [`lbug`](https://crates.io/crates/lbug) crate, so it requires
> `cmake` and a C++ toolchain at build time. It is the same engine consumers like
> Simard already depend on.

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

`CognitiveMemory` is scoped to a single agent and exposes a `store_*` method for
each of the six memory types. Every `store_*` call returns the new node id
(`Result<String>`); the matching reader / search methods (`get_sensory`,
`get_working`, `get_episodes`, `search_facts`, `search_procedures`,
`check_triggers`, …) read the stored data back.

```rust
use amplihack_memory::CognitiveMemory;

let mut memory = CognitiveMemory::new("checkout-agent")?;

// Sensory — raw short-lived observations.
// (modality, raw_data, ttl_seconds)
memory.store_sensory("visual", "User clicked submit button", 30)?;

// Working — bounded active-task context.
// (slot_type, content, task_id, relevance)
memory.store_working("task-context", "Processing checkout flow", "task-42", 0.95)?;

// Episodic — autobiographical event records.
// (content, source_label, temporal_index, metadata)
memory.store_episode("Resolved cart conflict by merging", "checkout-agent", None, None)?;

// Semantic — long-lived knowledge facts.
// (concept, content, confidence, source_id, tags, metadata)
let tags = ["checkout".to_string(), "conflict-resolution".to_string()];
memory.store_fact(
    "cart-merge",
    "When items conflict, prefer the newer addition",
    0.88,
    "exp_20250101_120000_abc123",
    Some(&tags),
    None,
)?;

// Procedural — reusable step-by-step procedures (idempotent by name).
// (name, steps, prerequisites)
let steps = ["Compare timestamps".to_string(), "Keep newer item".to_string()];
let prereqs = ["cart-access".to_string()];
memory.store_procedure("resolve-cart-conflict", &steps, Some(&prereqs))?;

// Prospective — future trigger → action pairs.
// (description, trigger_condition, action_on_trigger, priority)
memory.store_prospective(
    "Alert on cart abandonment",
    "no_activity 15min",
    "Send reminder email",
    1,
)?;

// Recall: keyword search over stored facts (query, limit, min_confidence).
let facts = memory.search_facts("cart", 10, 0.0);

// Fire any prospective memory whose trigger keywords appear in the content.
let fired = memory.check_triggers("no_activity detected on the checkout page");
```

### Persistent cognitive memory (`persistent` feature)

`CognitiveMemory` talks to its storage exclusively through the `GraphStore`
trait, so it is backend-pluggable. By default it uses an in-memory store; with
the `persistent` feature it can be backed by a durable, crash-safe
[LadybugDB](https://ladybugdb.com/) (Kùzu) database via the published `lbug`
crate — the same engine downstream consumers already use.

```rust
use amplihack_memory::CognitiveMemory;

// In-memory (default, unchanged): nothing is persisted.
let mut mem = CognitiveMemory::new("agent-1")?;

// Durable: data is stored at the given path and survives process restarts.
// (Requires the `persistent` feature.)
let mut mem = CognitiveMemory::open_persistent("/var/lib/agent/cognitive.ladybug", "agent-1")?;
mem.store_fact("rust", "memory safety without a GC", 0.9, "book", None, None)?;
mem.store_procedure("deploy", &["build".into(), "rollout".into()], None)?;
mem.checkpoint()?; // force-flush the WAL into the main DB file (bounded crash loss)
drop(mem);         // also checkpoints on drop

// Reopen later — facts, procedures, episodes and triggers are all still there.
// This transparently recovers from a WAL left corrupt by an unclean shutdown.
let mem = CognitiveMemory::open_persistent("/var/lib/agent/cognitive.ladybug", "agent-1")?;
assert_eq!(mem.search_facts("safety", 10, 0.0).len(), 1);
```

You can also plug in any `GraphStore` implementation directly:

```rust,ignore
let store = Box::new(MyCustomGraphStore::new());
let mem = CognitiveMemory::with_store("agent-1", store)?;
```

The persistent backend (`LbugGraphStore`) provides per-write `fsync` durability
barriers, crash-atomic multi-statement transactions, reopen-safe schema
introspection, agent isolation, and the corrected retrieval semantics
(tokenized multi-word recall, keyword-overlap trigger matching, idempotent
procedure storage). See `src/graph/lbug_store/` for details.

#### Durability & crash recovery

The persistent backend is hardened against the failure mode where a process is
killed mid-write and the write-ahead log (WAL) is left partially written:

- **Corrupt-WAL recovery.** `open_persistent` delegates to
  `open_persistent_with_recovery`, which first attempts a strict open; if the WAL
  cannot be replayed (the failure that previously made the store permanently
  unopenable and crashed with a C++ assertion), the unreplayable WAL is **moved
  aside** to `<wal>.corrupt-<timestamp>` (never deleted), the recoverable prefix
  is replayed, and a checkpoint folds it into the main database file. A
  structured `warn!` reports how many records survived. A clean open is
  unaffected: no artifact is written and no warning emitted.
- **Checkpoint API.** `CognitiveMemory::checkpoint()` forces the WAL into the
  main database file, so a subsequent clean reopen needs no replay. It is a
  no-op for the in-memory backend.
- **Bounded loss / auto-checkpoint.** The store auto-checkpoints after every
  `AUTO_CHECKPOINT_WRITES` (128) mutating operations and always on `close` /
  `Drop`, and leaves LadybugDB's own `auto_checkpoint` enabled to bound the WAL
  as a safety net. An unclean shutdown therefore strands at most a small,
  bounded number of writes in the WAL rather than every uncheckpointed record.

### Automatic `SIMILAR_TO` linking between facts

`CognitiveMemory` can automatically connect related semantic facts with
`SIMILAR_TO` edges, so knowledge about the same topic becomes traversable
without manual bookkeeping. Similarity is computed with the crate's existing
deterministic helpers — no embeddings, no network calls: a composite **Jaccard**
score that blends word overlap (0.5), tag overlap (0.2), and concept overlap
(0.3). The word and concept components are tokenized first — lower-cased, with
English stop words and tokens of two characters or fewer removed — so only
meaningful words drive the score. An edge is created when that score is **at or
above** a configurable threshold (default `0.60`).

Linking is **opt-in and additive** — `store_fact` and
`store_fact_with_provenance` are unchanged and never create `SIMILAR_TO` edges
on their own. You drive it explicitly with `auto_link_similar_facts`, in bulk
with `rebuild_similarity_links`, or at store time with `store_fact_with_options`.
Linking is strictly **same-agent** and **idempotent** — existing edges are never
duplicated.

`SimilarityOptions` controls the behaviour (it is `Copy`; pass
`&SimilarityOptions::default()` for the defaults):

| Field | Type | Default | Meaning |
|-------|------|---------|---------|
| `enabled` | `bool` | `true` | Master switch. `false` makes every entry point an inert no-op. |
| `threshold` | `f64` | `0.60` | Inclusive composite score (`>=`) required to create an edge. |
| `candidate_limit` | `usize` | `50` | Max other same-agent facts scored per source fact, taken highest-confidence first. |
| `bidirectional` | `bool` | `true` | Also create the reciprocal `B → A` edge. |

```rust
use amplihack_memory::{CognitiveMemory, SimilarityOptions, StoreFactOptions};

let mut mem = CognitiveMemory::new("research-agent")?;

let rust_tags = ["rust".to_string()];
let a = mem.store_fact("rust-safety",
    "the rust borrow checker guarantees memory safety", 0.9, "s1", Some(&rust_tags), None)?;
let _b = mem.store_fact("rust-safety",
    "rust enforces memory safety at compile time via its borrow checker", 0.85, "s2", Some(&rust_tags), None)?;
let _c = mem.store_fact("cooking",
    "simmer onions in butter until soft", 0.8, "s3", Some(&["food".to_string()]), None)?;

// Explicit: link `a` to every fact at/above the threshold (here: `b`, not `c`).
// Returns the number of NEW unordered pairs linked this call.
let linked = mem.auto_link_similar_facts(&a, &SimilarityOptions::default())?;
assert_eq!(linked, 1);

// Idempotent: a second call creates nothing new.
assert_eq!(mem.auto_link_similar_facts(&a, &SimilarityOptions::default())?, 0);

// Backfill / maintenance across the whole agent (additive, non-destructive).
let report = mem.rebuild_similarity_links(&SimilarityOptions::default())?;
println!("processed {} facts, {} new links", report.facts_processed, report.links_created);

// Opt-in store-time linking: wire up each new fact as it lands.
let opts = StoreFactOptions { similarity: Some(SimilarityOptions::default()) };
let _id = mem.store_fact_with_options("rust-safety",
    "memory safety in rust comes from ownership and the borrow checker",
    0.88, "s4", Some(&rust_tags), None, &[], &opts)?;
```

Each auto-created edge is `SemanticMemory --SIMILAR_TO--> SemanticMemory` between
two facts of the same agent, carrying a single `similarity_score` property — the
score as a fixed 4-decimal string (e.g. `"0.7321"`), parseable with
`parse::<f64>()`. With `StoreFactOptions::default()` (`similarity: None`),
`store_fact_with_options` is a byte-for-byte drop-in for
`store_fact_with_provenance` and creates no edges. On the `persistent` backend,
`SIMILAR_TO` edges and their scores are durable — they survive `checkpoint()`,
`drop`, and reopen.

This feature is **write-only for now**: it creates `SIMILAR_TO` edges but does
not yet add a public API to read them back (the graph is a private field; a
`similar_facts` reader is a deliberate, backward-compatible future addition).

See [`docs/similarity_linking.md`](docs/similarity_linking.md) for the full
reference, threshold-tuning guidance, directionality semantics, and a worked
tutorial.

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
from amplihack_memory_rs import MemoryConnector, Experience

# Create a connector (agent_name, db_path)
conn = MemoryConnector("my-agent", "/tmp/agent-memory")

# Store an experience
exp = Experience("success", "Parsed query", "Found 3 slots", 0.92)
exp_id = conn.store_experience(exp)
print(f"Stored: {exp_id}")

# Search (query, experience_type, min_confidence, limit)
results = conn.search("query parsing", None, 0.5, 10)
for r in results:
    print(f"  [{r.confidence:.0%}] {r.context}")
```

### Cognitive memory and pattern detection

```python
from amplihack_memory_rs import CognitiveMemory, PatternDetector

memory = CognitiveMemory("my-agent")
# Store items in each of the six memory types via record_sensory(),
# push_working(), store_episode(), store_fact(), store_procedure(), and
# store_prospective(); read them back with the matching get_recent_sensory(),
# recall_working(), recall_episodes(), search_facts(), recall_procedures(),
# and check_triggers() methods.

detector = PatternDetector(threshold=3, min_confidence=0.6)
# Feed experiences via add_discovery(), then query recognised patterns.
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
| `cognitive_memory`     | Six-type cognitive memory over a pluggable `GraphStore` (in-memory by default; durable LadybugDB via the `persistent` feature); optional automatic `SIMILAR_TO` linking between related facts |
| `connector`            | `MemoryConnector` — factory for backend lifecycle management     |
| `contradiction`        | Contradiction detection between semantic facts                   |
| `entity_extraction`    | Entity-name extraction from free text                            |
| `experience`           | `Experience` and `ExperienceType` data model                     |
| `graph`                | `GraphStore` trait with `InMemoryGraphStore`, `HiveGraphStore`, `FederatedGraphStore`, `KuzuGraphStore`, `LbugGraphStore` (feat: `persistent`) |
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
4. **Feature matrix** — test with `--features kuzu`, `--features ladybug`,
   `--features persistent`, and default features separately.

See [`AGENTS.md`](../../AGENTS.md) in the repository root for full contributor
guidelines.

## License

MIT
