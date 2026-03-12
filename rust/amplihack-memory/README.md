# amplihack-memory

Standalone memory system for goal-seeking agents. Provides persistent, queryable
memory with cognitive modeling, hierarchical organization, and graph-based
knowledge storage.

## Quick Start

```rust
use amplihack_memory::{MemoryConnector, Experience, ExperienceType};
use std::path::Path;

fn main() -> amplihack_memory::Result<()> {
    // Create a connector with 256 MB storage limit
    let mut connector = MemoryConnector::new(
        "my-agent",
        Some(Path::new("/tmp/agent-memory")),
        256,
    )?;

    // Store an experience
    let experience = Experience::new(
        ExperienceType::Success,
        "Parsed user query",
        "Extracted 3 intent slots",
        0.92,
    );
    let id = connector.store_experience(&experience)?;
    println!("Stored experience: {id}");

    Ok(())
}
```

## Features

- **Cognitive memory** — models agent experiences with confidence scoring
- **Hierarchical memory** — organizes knowledge across abstraction levels
- **Graph stores** — connects related experiences for associative recall
- **Security** — SHA-256 integrity checks on stored data
- **Pattern recognition** — identifies recurring patterns across experiences

## License

MIT
