use criterion::{black_box, criterion_group, criterion_main, Criterion};

use amplihack_memory::{
    compute_word_similarity, detect_contradiction, extract_entity_name, CognitiveMemory,
    Experience, ExperienceStore, ExperienceType, SemanticSearchEngine,
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
    let mut group = c.benchmark_group("semantic/search");
    for count in [100] {
        let docs: Vec<Experience> = (0..count).map(make_experience).collect();
        let engine = SemanticSearchEngine::new(docs);
        group.bench_function(format!("over_{count}_docs"), |b| {
            b.iter(|| engine.search(black_box("experience outcome details"), 10))
        });
    }
    group.finish();
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

criterion_group!(store_benches, bench_store_experiences, bench_store_search,);

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

criterion_main!(
    similarity_benches,
    extraction_benches,
    contradiction_benches,
    store_benches,
    search_benches,
    cognitive_benches,
);
