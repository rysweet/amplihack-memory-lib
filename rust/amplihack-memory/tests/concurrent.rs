//! Concurrent access tests for amplihack-memory.
//!
//! Verifies thread safety of shared state under concurrent reads, writes,
//! mixed operations, search, experience storage, and lock contention.

use std::collections::HashMap;
use std::sync::{Arc, Barrier, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use amplihack_memory::{
    Direction, Experience, ExperienceType, GraphStore, HierarchicalMemory, InMemoryGraphStore,
    SemanticSearchEngine,
};

const NUM_THREADS: usize = 8;
const OPS_PER_THREAD: usize = 100;

// ---------------------------------------------------------------------------
// 1. Concurrent graph store reads
// ---------------------------------------------------------------------------

#[test]
fn test_concurrent_graph_reads() {
    let store = Arc::new(Mutex::new(InMemoryGraphStore::new(Some("read-test"))));

    // Pre-populate nodes
    {
        let mut s = store.lock().unwrap();
        for i in 0..50 {
            let mut props = HashMap::new();
            props.insert("name".into(), format!("node-{i}"));
            s.add_node("Item", props, Some(&format!("n{i}"))).unwrap();
        }
    }

    let barrier = Arc::new(Barrier::new(NUM_THREADS));
    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|_| {
            let store = Arc::clone(&store);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                for i in 0..OPS_PER_THREAD {
                    let s = store.lock().unwrap();
                    let key = format!("n{}", i % 50);
                    let node = s.get_node(&key);
                    assert!(node.is_some(), "node {key} should exist");
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
}

// ---------------------------------------------------------------------------
// 2. Concurrent graph store writes
// ---------------------------------------------------------------------------

#[test]
fn test_concurrent_graph_writes() {
    let store = Arc::new(Mutex::new(InMemoryGraphStore::new(Some("write-test"))));

    let barrier = Arc::new(Barrier::new(NUM_THREADS));
    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|tid| {
            let store = Arc::clone(&store);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                for i in 0..OPS_PER_THREAD {
                    let mut s = store.lock().unwrap();
                    let node_id = format!("t{tid}_n{i}");
                    let mut props = HashMap::new();
                    props.insert("thread".into(), tid.to_string());
                    props.insert("idx".into(), i.to_string());
                    s.add_node("Item", props, Some(&node_id)).unwrap();
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    // Verify all nodes were created
    let s = store.lock().unwrap();
    let total = NUM_THREADS * OPS_PER_THREAD;
    let all_nodes = s.query_nodes("Item", None, total + 1);
    assert_eq!(
        all_nodes.len(),
        total,
        "expected {total} nodes, got {}",
        all_nodes.len()
    );
}

// ---------------------------------------------------------------------------
// 3. Mixed read/write concurrency
// ---------------------------------------------------------------------------

#[test]
fn test_concurrent_mixed_read_write() {
    let store = Arc::new(Mutex::new(InMemoryGraphStore::new(Some("mixed-test"))));

    // Seed some nodes so readers always find something
    {
        let mut s = store.lock().unwrap();
        for i in 0..10 {
            s.add_node("Seed", HashMap::new(), Some(&format!("seed{i}")))
                .unwrap();
        }
    }

    let barrier = Arc::new(Barrier::new(NUM_THREADS));
    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|tid| {
            let store = Arc::clone(&store);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                for i in 0..OPS_PER_THREAD {
                    if tid % 2 == 0 {
                        // Writer thread: add nodes + edges
                        let mut s = store.lock().unwrap();
                        let node_id = format!("w{tid}_n{i}");
                        s.add_node("Dynamic", HashMap::new(), Some(&node_id))
                            .unwrap();
                        let seed = format!("seed{}", i % 10);
                        let _ = s.add_edge(&node_id, &seed, "REFS", None);
                    } else {
                        // Reader thread: query nodes and neighbors
                        let s = store.lock().unwrap();
                        let seed = format!("seed{}", i % 10);
                        let _ = s.get_node(&seed);
                        let _ = s.query_nodes("Seed", None, 100);
                        let _ = s.query_neighbors(&seed, None, Direction::Both, 50);
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    // Verify seed nodes are intact
    let s = store.lock().unwrap();
    for i in 0..10 {
        assert!(
            s.get_node(&format!("seed{i}")).is_some(),
            "seed node {i} should survive concurrent access"
        );
    }
}

// ---------------------------------------------------------------------------
// 4. Concurrent search under contention
// ---------------------------------------------------------------------------

#[test]
fn test_concurrent_search() {
    let store = Arc::new(Mutex::new(InMemoryGraphStore::new(Some("search-test"))));

    // Populate with searchable content
    {
        let mut s = store.lock().unwrap();
        let topics = [
            "rust programming language systems",
            "python machine learning data",
            "graph database knowledge storage",
            "concurrent thread safety mutex",
            "memory hierarchy cognitive agents",
        ];
        for (i, topic) in topics.iter().enumerate() {
            let mut props = HashMap::new();
            props.insert("content".into(), topic.to_string());
            s.add_node("Fact", props, Some(&format!("fact{i}")))
                .unwrap();
        }
    }

    let queries = ["rust", "memory", "graph", "python", "thread"];
    let barrier = Arc::new(Barrier::new(NUM_THREADS));
    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|tid| {
            let store = Arc::clone(&store);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                for i in 0..OPS_PER_THREAD {
                    let s = store.lock().unwrap();
                    let query = queries[i % queries.len()];
                    let results = s.search_nodes("Fact", &["content".to_string()], query, None, 10);
                    assert!(
                        !results.is_empty(),
                        "thread {tid}: search for '{query}' returned no results"
                    );
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
}

// ---------------------------------------------------------------------------
// 5. Concurrent experience storage via HierarchicalMemory
// ---------------------------------------------------------------------------

#[test]
fn test_concurrent_hierarchical_memory() {
    let mem = Arc::new(Mutex::new(
        HierarchicalMemory::new("concurrent-agent").unwrap(),
    ));

    let barrier = Arc::new(Barrier::new(NUM_THREADS));
    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|tid| {
            let mem = Arc::clone(&mem);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                for i in 0..OPS_PER_THREAD {
                    let mut m = mem.lock().unwrap();
                    let content = format!("thread {tid} knowledge item {i} about testing");
                    let concept = format!("concept-t{tid}-{i}");
                    let confidence = 0.5 + (i as f64 / OPS_PER_THREAD as f64) * 0.5;
                    m.store_knowledge(
                        &content,
                        &concept,
                        confidence,
                        None,
                        &format!("source-{tid}"),
                        &[format!("tag-{tid}")],
                        None,
                    )
                    .unwrap();
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    // Verify total knowledge stored
    let m = mem.lock().unwrap();
    let stats = m.get_statistics();
    let total_nodes = stats
        .get("semantic_nodes")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let expected = NUM_THREADS * OPS_PER_THREAD;
    assert_eq!(
        total_nodes, expected,
        "expected {expected} knowledge nodes, got {total_nodes}"
    );
}

// ---------------------------------------------------------------------------
// 6. Lock contention / no deadlock (with timeout)
// ---------------------------------------------------------------------------

#[test]
fn test_no_deadlock_under_contention() {
    let store = Arc::new(Mutex::new(InMemoryGraphStore::new(Some("deadlock-test"))));

    // Seed nodes and edges
    {
        let mut s = store.lock().unwrap();
        for i in 0..20 {
            s.add_node("N", HashMap::new(), Some(&format!("dl{i}")))
                .unwrap();
        }
        for i in 0..19 {
            s.add_edge(&format!("dl{i}"), &format!("dl{}", i + 1), "NEXT", None)
                .unwrap();
        }
    }

    let deadline = Instant::now() + Duration::from_secs(30);
    let barrier = Arc::new(Barrier::new(NUM_THREADS));
    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|tid| {
            let store = Arc::clone(&store);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                for i in 0..OPS_PER_THREAD {
                    assert!(
                        Instant::now() < deadline,
                        "thread {tid} timed out — possible deadlock"
                    );

                    let mut s = store.lock().unwrap();
                    match i % 5 {
                        0 => {
                            let id = format!("extra-{tid}-{i}");
                            s.add_node("N", HashMap::new(), Some(&id)).unwrap();
                        }
                        1 => {
                            let _ = s.get_node(&format!("dl{}", i % 20));
                        }
                        2 => {
                            let _ = s.query_nodes("N", None, 50);
                        }
                        3 => {
                            let _ = s.traverse("dl0", None, 3, Direction::Outgoing, None);
                        }
                        4 => {
                            let mut props = HashMap::new();
                            props.insert("updated_by".into(), format!("t{tid}"));
                            let _ = s.update_node(&format!("dl{}", i % 20), props);
                        }
                        _ => unreachable!(),
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join()
            .expect("thread panicked — possible deadlock or data race");
    }
}

// ---------------------------------------------------------------------------
// 7. Arc<Mutex> correctness — data consistency after concurrent modifications
// ---------------------------------------------------------------------------

#[test]
fn test_arc_mutex_data_consistency() {
    let store = Arc::new(Mutex::new(InMemoryGraphStore::new(Some(
        "consistency-test",
    ))));

    let barrier = Arc::new(Barrier::new(NUM_THREADS));
    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|tid| {
            let store = Arc::clone(&store);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                for i in 0..OPS_PER_THREAD {
                    let mut s = store.lock().unwrap();

                    // Each thread adds a node then immediately verifies it exists
                    let id = format!("c{tid}_{i}");
                    let mut props = HashMap::new();
                    props.insert("value".into(), format!("{}", tid * 1000 + i));
                    s.add_node("Counter", props, Some(&id)).unwrap();

                    let node = s.get_node(&id);
                    assert!(node.is_some(), "just-inserted node {id} must exist");
                    assert_eq!(
                        node.unwrap().properties.get("value").unwrap(),
                        &format!("{}", tid * 1000 + i),
                        "property mismatch for node {id}"
                    );
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    // Final consistency check: every node present with correct values
    let s = store.lock().unwrap();
    for tid in 0..NUM_THREADS {
        for i in 0..OPS_PER_THREAD {
            let id = format!("c{tid}_{i}");
            let node = s.get_node(&id);
            assert!(
                node.is_some(),
                "node {id} missing after all threads finished"
            );
            let val = node.unwrap().properties.get("value").unwrap().clone();
            assert_eq!(
                val,
                format!("{}", tid * 1000 + i),
                "corrupted value for {id}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 8. Concurrent SemanticSearchEngine access
// ---------------------------------------------------------------------------

#[test]
fn test_concurrent_semantic_search_engine() {
    let engine = Arc::new(Mutex::new(SemanticSearchEngine::new(Vec::new())));

    let barrier = Arc::new(Barrier::new(NUM_THREADS));
    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|tid| {
            let engine = Arc::clone(&engine);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                for i in 0..OPS_PER_THREAD {
                    if tid % 2 == 0 {
                        // Writer: add experiences
                        let exp = Experience::new(
                            ExperienceType::Success,
                            format!("thread {tid} task {i} completed build"),
                            format!("outcome {i} from thread {tid}"),
                            0.8,
                        )
                        .unwrap();
                        let mut e = engine.lock().unwrap();
                        e.add_experience(exp);
                    } else {
                        // Reader: search
                        let e = engine.lock().unwrap();
                        let _ = e.search("build", 5);
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    // Verify corpus populated by writer threads
    let e = engine.lock().unwrap();
    let writer_threads = NUM_THREADS / 2;
    let expected = writer_threads * OPS_PER_THREAD;
    assert_eq!(
        e.corpus_size(),
        expected,
        "expected {expected} experiences, got {}",
        e.corpus_size()
    );
}

// ---------------------------------------------------------------------------
// 9. Concurrent SqliteBackend access
// ---------------------------------------------------------------------------

#[test]
fn test_concurrent_sqlite_backend() {
    use amplihack_memory::{ExperienceBackend, SqliteBackend};
    use tempfile::TempDir;

    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("concurrent_test.db");
    let backend = SqliteBackend::new(&db_path, "concurrent-agent", 100, true).unwrap();
    let backend = Arc::new(Mutex::new(backend));

    let threads = 4;
    let ops = 25;
    let barrier = Arc::new(Barrier::new(threads));
    let handles: Vec<_> = (0..threads)
        .map(|tid| {
            let backend = Arc::clone(&backend);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                for i in 0..ops {
                    let exp = Experience::new(
                        ExperienceType::Success,
                        format!("thread {tid} context {i}"),
                        format!("thread {tid} outcome {i}"),
                        0.8,
                    )
                    .unwrap();
                    let mut b = backend.lock().unwrap();
                    b.add(&exp).unwrap();
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let b = backend.lock().unwrap();
    let stats = b.get_statistics().unwrap();
    assert_eq!(
        stats.total_experiences as usize,
        threads * ops,
        "expected {} total experiences, got {}",
        threads * ops,
        stats.total_experiences
    );
}
