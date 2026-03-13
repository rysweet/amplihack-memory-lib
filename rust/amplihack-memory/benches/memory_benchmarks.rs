use std::collections::HashMap;

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use amplihack_memory::{
    compute_word_similarity, detect_contradiction, extract_entity_name, CognitiveMemory, Direction,
    Experience, ExperienceStore, ExperienceType, GraphStore, HierarchicalMemory,
    InMemoryGraphStore, PatternDetector, SemanticSearchEngine,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn words(n: usize) -> String {
    let base = [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "alpha", "beta", "gamma",
        "delta", "epsilon", "zeta", "compiler", "runtime", "module", "function", "variable",
        "constant",
    ];
    (0..n)
        .map(|i| base[i % base.len()])
        .collect::<Vec<_>>()
        .join(" ")
}

fn entity_text(count: usize) -> String {
    let names = [
        "Alice Smith",
        "Bob Johnson",
        "Carol Williams",
        "David Brown",
        "Eve Davis",
        "Frank Miller",
        "Grace Wilson",
        "Henry Moore",
        "Iris Taylor",
        "Jack Anderson",
        "Karen Thomas",
        "Leo Jackson",
        "Maria White",
        "Noah Harris",
        "Olivia Martin",
        "Paul Garcia",
        "Quinn Martinez",
        "Rosa Robinson",
        "Sam Clark",
        "Tina Rodriguez",
    ];
    let mut parts = Vec::with_capacity(count);
    for i in 0..count {
        parts.push(format!(
            "Met with {} at the office.",
            names[i % names.len()]
        ));
    }
    parts.join(" ")
}

fn make_experience(i: usize) -> Experience {
    Experience::new(
        ExperienceType::Success,
        format!("context for experience number {i}"),
        format!("outcome details for item {i} with extra words"),
        0.8,
    )
    .unwrap()
}

// ---------------------------------------------------------------------------
// Benchmark 1: Similarity computation
// ---------------------------------------------------------------------------

fn bench_similarity_small(c: &mut Criterion) {
    let a = words(10);
    let b = words(10);
    c.bench_function("similarity/word_10", |b_iter| {
        b_iter.iter(|| compute_word_similarity(black_box(&a), black_box(&b)))
    });
}

fn bench_similarity_medium(c: &mut Criterion) {
    let a = words(100);
    let b = words(100);
    c.bench_function("similarity/word_100", |b_iter| {
        b_iter.iter(|| compute_word_similarity(black_box(&a), black_box(&b)))
    });
}

fn bench_similarity_large(c: &mut Criterion) {
    let a = words(1000);
    let b = words(1000);
    c.bench_function("similarity/word_1000", |b_iter| {
        b_iter.iter(|| compute_word_similarity(black_box(&a), black_box(&b)))
    });
}

// ---------------------------------------------------------------------------
// Benchmark 2: Entity extraction
// ---------------------------------------------------------------------------

fn bench_extract_simple(c: &mut Criterion) {
    let text = entity_text(1);
    c.bench_function("extraction/1_entity", |b| {
        b.iter(|| extract_entity_name(black_box(&text), black_box("")))
    });
}

fn bench_extract_moderate(c: &mut Criterion) {
    let text = entity_text(5);
    c.bench_function("extraction/5_entities", |b| {
        b.iter(|| extract_entity_name(black_box(&text), black_box("")))
    });
}

fn bench_extract_complex(c: &mut Criterion) {
    let text = entity_text(20);
    c.bench_function("extraction/20_entities", |b| {
        b.iter(|| extract_entity_name(black_box(&text), black_box("")))
    });
}

// ---------------------------------------------------------------------------
// Benchmark 3: Contradiction detection
// ---------------------------------------------------------------------------

fn bench_contradiction_similar(c: &mut Criterion) {
    c.bench_function("contradiction/similar_pair", |b| {
        b.iter(|| {
            detect_contradiction(
                black_box("The team has 5 members"),
                black_box("The team has 8 members"),
                black_box("team size"),
                black_box("team size"),
            )
        })
    });
}

fn bench_contradiction_dissimilar(c: &mut Criterion) {
    c.bench_function("contradiction/dissimilar_pair", |b| {
        b.iter(|| {
            detect_contradiction(
                black_box("The sky is blue today"),
                black_box("Rust compiles to native code"),
                black_box("weather"),
                black_box("programming"),
            )
        })
    });
}

fn bench_contradiction_no_numbers(c: &mut Criterion) {
    c.bench_function("contradiction/no_numbers", |b| {
        b.iter(|| {
            detect_contradiction(
                black_box("The project uses Rust"),
                black_box("The project uses Python"),
                black_box("language"),
                black_box("language"),
            )
        })
    });
}

// ---------------------------------------------------------------------------
// Benchmark 4: Experience store/retrieve cycle
// ---------------------------------------------------------------------------

fn bench_store_experiences(c: &mut Criterion) {
    let mut group = c.benchmark_group("store/add");
    for count in [1, 10, 100] {
        let experiences: Vec<Experience> = (0..count).map(make_experience).collect();
        group.bench_function(format!("{count}"), |b| {
            b.iter_with_setup(
                || {
                    let tmp = tempfile::tempdir().unwrap();
                    let store = ExperienceStore::new(
                        "bench-agent",
                        true,
                        None,
                        None,
                        100,
                        Some(tmp.path()),
                    )
                    .unwrap();
                    (store, tmp, experiences.clone())
                },
                |(mut store, _tmp, exps)| {
                    for exp in &exps {
                        store.add(black_box(exp)).unwrap();
                    }
                },
            );
        });
    }
    group.finish();
}

fn bench_store_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("store/search");
    for count in [100, 1000] {
        group.bench_function(format!("over_{count}"), |b| {
            b.iter_with_setup(
                || {
                    let tmp = tempfile::tempdir().unwrap();
                    let mut store = ExperienceStore::new(
                        "bench-agent",
                        true,
                        None,
                        None,
                        200,
                        Some(tmp.path()),
                    )
                    .unwrap();
                    for i in 0..count {
                        store.add(&make_experience(i)).unwrap();
                    }
                    (store, tmp)
                },
                |(store, _tmp)| {
                    store
                        .search(black_box("experience"), None, 0.0, 10)
                        .unwrap();
                },
            );
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark 5: Semantic search
// ---------------------------------------------------------------------------

fn bench_semantic_indexing(c: &mut Criterion) {
    let mut group = c.benchmark_group("semantic/index");
    for count in [10, 100] {
        let docs: Vec<Experience> = (0..count).map(make_experience).collect();
        group.bench_function(format!("{count}_docs"), |b| {
            b.iter(|| {
                let mut engine = SemanticSearchEngine::new(vec![]);
                for doc in docs.iter() {
                    engine.add_experience(black_box(doc.clone()));
                }
                engine
            })
        });
    }
    group.finish();
}

fn bench_semantic_search(c: &mut Criterion) {
    let count = 100;
    let docs: Vec<Experience> = (0..count).map(make_experience).collect();
    let engine = SemanticSearchEngine::new(docs);
    c.bench_function("semantic/search/over_100_docs", |b| {
        b.iter(|| engine.search(black_box("experience outcome details"), 10))
    });
}

// ---------------------------------------------------------------------------
// Benchmark 6: Cognitive memory store/recall
// ---------------------------------------------------------------------------

fn bench_cognitive_store_fact(c: &mut Criterion) {
    c.bench_function("cognitive/store_fact", |b| {
        b.iter_with_setup(
            || CognitiveMemory::new("bench-agent").unwrap(),
            |mut cm| {
                cm.store_fact(
                    black_box("rust"),
                    black_box("Rust is a systems language"),
                    0.95,
                    "bench",
                    Some(&["lang".into()]),
                    None,
                )
                .unwrap();
                cm.close();
            },
        )
    });
}

fn bench_cognitive_search_facts(c: &mut Criterion) {
    c.bench_function("cognitive/search_facts", |b| {
        b.iter_with_setup(
            || {
                let mut cm = CognitiveMemory::new("bench-agent").unwrap();
                for i in 0..50 {
                    cm.store_fact(
                        &format!("concept_{i}"),
                        &format!("Fact number {i} about programming and systems"),
                        0.9,
                        "bench",
                        None,
                        None,
                    )
                    .unwrap();
                }
                cm
            },
            |cm| {
                let _results = cm.search_facts(black_box("programming"), 10, 0.0);
            },
        )
    });
}

fn bench_cognitive_store_episode(c: &mut Criterion) {
    c.bench_function("cognitive/store_episode", |b| {
        b.iter_with_setup(
            || CognitiveMemory::new("bench-agent").unwrap(),
            |mut cm| {
                cm.store_episode(
                    black_box("Completed the Rust tutorial"),
                    "learning",
                    None,
                    None,
                )
                .unwrap();
                cm.close();
            },
        )
    });
}

fn bench_cognitive_search_episodes(c: &mut Criterion) {
    c.bench_function("cognitive/search_episodes", |b| {
        b.iter_with_setup(
            || {
                let mut cm = CognitiveMemory::new("bench-agent").unwrap();
                for i in 0..50 {
                    cm.store_episode(
                        &format!("Episode {i}: worked on module {i}"),
                        "development",
                        None,
                        None,
                    )
                    .unwrap();
                }
                cm
            },
            |cm| {
                let _results = cm.search_episodes(10);
            },
        )
    });
}

// ---------------------------------------------------------------------------
// Benchmark 7: Graph operations
// ---------------------------------------------------------------------------

fn make_graph_store(num_nodes: usize) -> InMemoryGraphStore {
    let mut store = InMemoryGraphStore::new(Some("bench"));
    for i in 0..num_nodes {
        let mut props = HashMap::new();
        props.insert("name".into(), format!("node_{i}"));
        props.insert("description".into(), format!("description for node {i}"));
        store
            .add_node("Entity", props, Some(&format!("n{i}")))
            .unwrap();
    }
    store
}

fn bench_graph_add_nodes(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph/add_nodes");
    for count in [10, 100, 1000] {
        group.bench_function(format!("{count}"), |b| {
            b.iter(|| {
                let mut store = InMemoryGraphStore::new(Some("bench"));
                for i in 0..count {
                    let mut props = HashMap::new();
                    props.insert("name".into(), format!("node_{i}"));
                    store.add_node("Entity", black_box(props), None).unwrap();
                }
                store
            })
        });
    }
    group.finish();
}

fn bench_graph_add_edges(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph/add_edges");
    for count in [10, 100, 500] {
        group.bench_function(format!("{count}"), |b| {
            b.iter_with_setup(
                || make_graph_store(count),
                |mut store| {
                    for i in 0..count - 1 {
                        store
                            .add_edge(
                                &format!("n{i}"),
                                &format!("n{}", i + 1),
                                black_box("RELATES_TO"),
                                None,
                            )
                            .unwrap();
                    }
                },
            )
        });
    }
    group.finish();
}

fn bench_graph_search_nodes(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph/search_nodes");
    for count in [100, 500, 1000] {
        let store = make_graph_store(count);
        group.bench_function(format!("over_{count}"), |b| {
            b.iter(|| {
                store.search_nodes(
                    "Entity",
                    &["name".into(), "description".into()],
                    black_box("node_50"),
                    None,
                    10,
                )
            })
        });
    }
    group.finish();
}

fn bench_graph_query_neighbors(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph/query_neighbors");
    let mut store = make_graph_store(200);
    for i in 0..199 {
        store
            .add_edge(&format!("n{i}"), &format!("n{}", i + 1), "NEXT", None)
            .unwrap();
    }
    for depth in [1, 2, 3] {
        group.bench_function(format!("depth_{depth}"), |b| {
            b.iter(|| {
                store.traverse(
                    black_box("n0"),
                    Some(&["NEXT".into()]),
                    depth,
                    Direction::Outgoing,
                    None,
                )
            })
        });
    }
    group.finish();
}

fn bench_graph_incoming_edges(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph/incoming_edges");
    for count in [50, 200] {
        let mut store = make_graph_store(count + 1);
        for i in 0..count {
            store
                .add_edge(&format!("n{i}"), &format!("n{count}"), "POINTS_TO", None)
                .unwrap();
        }
        group.bench_function(format!("{count}_incoming"), |b| {
            b.iter(|| {
                store.query_neighbors(
                    black_box(&format!("n{count}")),
                    Some("POINTS_TO"),
                    Direction::Incoming,
                    count,
                )
            })
        });
    }
    group.finish();
}

fn bench_graph_delete_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph/delete");
    group.bench_function("nodes_50", |b| {
        b.iter_with_setup(
            || {
                let mut store = make_graph_store(50);
                for i in 0..49 {
                    store
                        .add_edge(&format!("n{i}"), &format!("n{}", i + 1), "LINKED", None)
                        .unwrap();
                }
                store
            },
            |mut store| {
                for i in 0..50 {
                    let _ = store.delete_node(black_box(&format!("n{i}")));
                }
            },
        )
    });
    group.bench_function("edges_50", |b| {
        b.iter_with_setup(
            || {
                let mut store = make_graph_store(51);
                for i in 0..50 {
                    store
                        .add_edge(&format!("n{i}"), &format!("n{}", i + 1), "LINKED", None)
                        .unwrap();
                }
                store
            },
            |mut store| {
                for i in 0..50 {
                    let _ = store.delete_edge(
                        black_box(&format!("n{i}")),
                        black_box(&format!("n{}", i + 1)),
                        "LINKED",
                    );
                }
            },
        )
    });
    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark 8: Hierarchical memory
// ---------------------------------------------------------------------------

fn make_hierarchical_memory(num_entries: usize) -> HierarchicalMemory {
    let mut hm = HierarchicalMemory::new("bench-agent").unwrap();
    for i in 0..num_entries {
        hm.store_knowledge(
            &format!("Knowledge fact number {i} about software engineering"),
            &format!("concept_{i}"),
            0.85,
            None,
            "bench",
            &["test".into()],
            None,
        )
        .unwrap();
    }
    hm
}

fn bench_hierarchical_store_knowledge(c: &mut Criterion) {
    let mut group = c.benchmark_group("hierarchical/store");
    for count in [1, 10, 50] {
        group.bench_function(format!("{count}_items"), |b| {
            b.iter_with_setup(
                || HierarchicalMemory::new("bench-agent").unwrap(),
                |mut hm| {
                    for i in 0..count {
                        hm.store_knowledge(
                            black_box(&format!("Knowledge fact {i} about distributed systems")),
                            &format!("concept_{i}"),
                            0.85,
                            None,
                            "bench",
                            &["test".into()],
                            None,
                        )
                        .unwrap();
                    }
                    hm.close();
                },
            )
        });
    }
    group.finish();
}

fn bench_hierarchical_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("hierarchical/search");
    for count in [20, 100] {
        let hm = make_hierarchical_memory(count);
        group.bench_function(format!("over_{count}"), |b| {
            b.iter(|| hm.retrieve_subgraph(black_box("software engineering"), 2, 10))
        });
    }
    group.finish();
}

fn bench_hierarchical_get_statistics(c: &mut Criterion) {
    let mut group = c.benchmark_group("hierarchical/statistics");
    for count in [20, 100] {
        let hm = make_hierarchical_memory(count);
        group.bench_function(format!("over_{count}"), |b| b.iter(|| hm.get_statistics()));
    }
    group.finish();
}

fn bench_hierarchical_consolidate(c: &mut Criterion) {
    let mut group = c.benchmark_group("hierarchical/consolidate");
    for count in [20, 100] {
        let hm = make_hierarchical_memory(count);
        group.bench_function(format!("aggregation_{count}"), |b| {
            b.iter(|| {
                hm.execute_aggregation(black_box("count"), "", 100);
                hm.execute_aggregation(black_box("avg_confidence"), "", 100);
                hm.execute_aggregation(black_box("top_concepts"), "", 10);
                hm.execute_aggregation(black_box("by_category"), "", 100);
            })
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark 9: Concurrent operations
// ---------------------------------------------------------------------------

fn bench_concurrent_reads(c: &mut Criterion) {
    c.bench_function("concurrent/reads_4_threads", |b| {
        b.iter_with_setup(
            || {
                let store = make_graph_store(500);
                std::sync::Arc::new(std::sync::Mutex::new(store))
            },
            |store| {
                std::thread::scope(|s| {
                    for t in 0..4 {
                        let store = std::sync::Arc::clone(&store);
                        s.spawn(move || {
                            let store = store.lock().unwrap();
                            for i in 0..50 {
                                let idx = (t * 50 + i) % 500;
                                black_box(store.get_node(&format!("n{idx}")));
                            }
                        });
                    }
                });
            },
        )
    });
}

fn bench_concurrent_writes(c: &mut Criterion) {
    c.bench_function("concurrent/writes_4_threads", |b| {
        b.iter_with_setup(
            || {
                let store = InMemoryGraphStore::new(Some("bench"));
                std::sync::Arc::new(std::sync::Mutex::new(store))
            },
            |store| {
                std::thread::scope(|s| {
                    for t in 0..4 {
                        let store = std::sync::Arc::clone(&store);
                        s.spawn(move || {
                            let mut store = store.lock().unwrap();
                            for i in 0..25 {
                                let mut props = HashMap::new();
                                props.insert("name".into(), format!("t{t}_node_{i}"));
                                store.add_node("Entity", props, None).unwrap();
                            }
                        });
                    }
                });
            },
        )
    });
}

fn bench_concurrent_mixed(c: &mut Criterion) {
    c.bench_function("concurrent/mixed_4_threads", |b| {
        b.iter_with_setup(
            || {
                let store = make_graph_store(200);
                std::sync::Arc::new(std::sync::Mutex::new(store))
            },
            |store| {
                std::thread::scope(|s| {
                    // 2 reader threads
                    for t in 0..2 {
                        let store = std::sync::Arc::clone(&store);
                        s.spawn(move || {
                            let store = store.lock().unwrap();
                            for i in 0..50 {
                                let idx = (t * 50 + i) % 200;
                                black_box(store.get_node(&format!("n{idx}")));
                            }
                        });
                    }
                    // 2 writer threads
                    for t in 0..2 {
                        let store = std::sync::Arc::clone(&store);
                        s.spawn(move || {
                            let mut store = store.lock().unwrap();
                            for i in 0..25 {
                                let mut props = HashMap::new();
                                props.insert("name".into(), format!("new_t{t}_{i}"));
                                store.add_node("Entity", props, None).unwrap();
                            }
                        });
                    }
                });
            },
        )
    });
}

// ---------------------------------------------------------------------------
// Benchmark 10: Pattern recognition
// ---------------------------------------------------------------------------

fn make_discoveries(n: usize) -> Vec<HashMap<String, serde_json::Value>> {
    let patterns = [
        "retry_success",
        "timeout_failure",
        "cache_hit",
        "rate_limit",
    ];
    (0..n)
        .map(|i| {
            let mut d = HashMap::new();
            d.insert(
                "pattern_type".into(),
                serde_json::json!(patterns[i % patterns.len()]),
            );
            d.insert("outcome".into(), serde_json::json!("observed"));
            d.insert("iteration".into(), serde_json::json!(i));
            d
        })
        .collect()
}

fn bench_pattern_detect(c: &mut Criterion) {
    let mut group = c.benchmark_group("pattern/detect");
    for count in [10, 50, 200] {
        let discoveries = make_discoveries(count);
        group.bench_function(format!("{count}_discoveries"), |b| {
            b.iter(|| {
                let mut detector = PatternDetector::new(3, 0.5);
                for d in &discoveries {
                    detector.add_discovery(black_box(d));
                }
                detector.get_recognized_patterns(None)
            })
        });
    }
    group.finish();
}

fn bench_pattern_validate(c: &mut Criterion) {
    let mut group = c.benchmark_group("pattern/validate");
    for count in [10, 50] {
        group.bench_function(format!("{count}_validations"), |b| {
            b.iter_with_setup(
                || {
                    let mut detector = PatternDetector::new(2, 0.3);
                    let discoveries = make_discoveries(20);
                    for d in &discoveries {
                        detector.add_discovery(d);
                    }
                    detector
                },
                |mut detector| {
                    let patterns = [
                        "retry_success",
                        "timeout_failure",
                        "cache_hit",
                        "rate_limit",
                    ];
                    for i in 0..count {
                        detector
                            .validate_pattern(black_box(patterns[i % patterns.len()]), i % 3 != 0);
                    }
                },
            )
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Groups
// ---------------------------------------------------------------------------

criterion_group!(
    similarity_benches,
    bench_similarity_small,
    bench_similarity_medium,
    bench_similarity_large,
);

criterion_group!(
    extraction_benches,
    bench_extract_simple,
    bench_extract_moderate,
    bench_extract_complex,
);

criterion_group!(
    contradiction_benches,
    bench_contradiction_similar,
    bench_contradiction_dissimilar,
    bench_contradiction_no_numbers,
);

criterion_group!(store_benches, bench_store_experiences, bench_store_search);

criterion_group!(
    search_benches,
    bench_semantic_indexing,
    bench_semantic_search,
);

criterion_group!(
    cognitive_benches,
    bench_cognitive_store_fact,
    bench_cognitive_search_facts,
    bench_cognitive_store_episode,
    bench_cognitive_search_episodes,
);

criterion_group!(
    graph_benches,
    bench_graph_add_nodes,
    bench_graph_add_edges,
    bench_graph_search_nodes,
    bench_graph_query_neighbors,
    bench_graph_incoming_edges,
    bench_graph_delete_operations,
);

criterion_group!(
    hierarchical_benches,
    bench_hierarchical_store_knowledge,
    bench_hierarchical_search,
    bench_hierarchical_get_statistics,
    bench_hierarchical_consolidate,
);

criterion_group!(
    concurrent_benches,
    bench_concurrent_reads,
    bench_concurrent_writes,
    bench_concurrent_mixed,
);

criterion_group!(
    pattern_benches,
    bench_pattern_detect,
    bench_pattern_validate,
);

criterion_main!(
    similarity_benches,
    extraction_benches,
    contradiction_benches,
    store_benches,
    search_benches,
    cognitive_benches,
    graph_benches,
    hierarchical_benches,
    concurrent_benches,
    pattern_benches,
);
