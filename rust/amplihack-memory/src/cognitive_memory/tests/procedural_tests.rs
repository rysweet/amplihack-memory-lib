use super::super::*;

fn make_cm() -> CognitiveMemory {
    CognitiveMemory::new(&format!("test-agent-{}", uuid::Uuid::new_v4())).unwrap()
}

// -- procedural memory --

#[test]
fn test_store_and_search_procedures() {
    let mut cm = make_cm();
    let steps = vec!["cargo build".into(), "cargo test".into()];
    let id = cm.store_procedure("build-rust", &steps, None).unwrap();
    assert!(id.starts_with("proc_"));

    let procs = cm.search_procedures("rust", 10);
    assert_eq!(procs.len(), 1);
    assert_eq!(procs[0].name, "build-rust");
    assert_eq!(procs[0].steps, steps);
}

#[test]
fn test_search_procedures_increments_usage() {
    let mut cm = make_cm();
    let steps = vec!["step1".into()];
    cm.store_procedure("deploy", &steps, None).unwrap();

    let procs = cm.search_procedures_mut("deploy", 10);
    assert_eq!(procs[0].usage_count, 0); // returned before increment

    // After search_procedures_mut, the stored count should be 1
    let procs2 = cm.search_procedures("deploy", 10);
    assert_eq!(procs2[0].usage_count, 1);
}

#[test]
fn test_procedure_with_prerequisites() {
    let mut cm = make_cm();
    let steps = vec!["run tests".into()];
    let prereqs = vec!["install deps".into()];
    cm.store_procedure("test-suite", &steps, Some(&prereqs))
        .unwrap();

    let procs = cm.search_procedures("test", 10);
    assert_eq!(procs[0].prerequisites, prereqs);
}

#[test]
fn test_search_procedures_empty_query_returns_all() {
    let mut cm = make_cm();
    cm.store_procedure("a", &["s".into()], None).unwrap();
    cm.store_procedure("b", &["s".into()], None).unwrap();

    let procs = cm.search_procedures("", 10);
    assert_eq!(procs.len(), 2);
}

// -- Extended coverage: search_procedures_mut --

#[test]
fn test_search_procedures_mut_multiple_increments() {
    let mut cm = make_cm();
    cm.store_procedure("build", &["compile".into()], None)
        .unwrap();

    cm.search_procedures_mut("build", 10);
    cm.search_procedures_mut("build", 10);
    cm.search_procedures_mut("build", 10);

    let procs = cm.search_procedures("build", 10);
    assert_eq!(procs[0].usage_count, 3);
}

// -- Corrected retrieval semantics (the consumer-fork bugs) --

#[test]
fn test_search_procedures_tokenized_multiword_or() {
    let mut cm = make_cm();
    cm.store_procedure(
        "release-pipeline",
        &[
            "compile sources".into(),
            "build container image".into(),
            "deploy".into(),
        ],
        None,
    )
    .unwrap();

    // Out-of-order, mixed-case, matched across name + steps via tokenized
    // CONTAINS (not a whole-string match).
    assert_eq!(
        cm.search_procedures("IMAGE compile", 10).len(),
        1,
        "tokenized procedure recall should match across name and steps"
    );

    // OR semantics: one overlapping token suffices.
    assert_eq!(
        cm.search_procedures("image nonexistent", 10).len(),
        1,
        "one overlapping token should match (OR per token)"
    );

    // No overlap -> no match.
    assert!(cm.search_procedures("kubernetes helm", 10).is_empty());
}

#[test]
fn test_store_procedure_is_idempotent_by_name() {
    let mut cm = make_cm();

    let id1 = cm
        .store_procedure("build", &["step-a".into()], None)
        .unwrap();
    let id2 = cm
        .store_procedure("build", &["step-a".into(), "step-b".into()], None)
        .unwrap();

    // Re-storing the same name reuses the node instead of inserting a duplicate.
    assert_eq!(id1, id2, "re-storing the same name must reuse the node");
    assert_eq!(
        cm.get_memory_stats().get("procedural"),
        Some(&1),
        "idempotent store must not accumulate duplicates"
    );

    // The second store updated the steps in place.
    let procs = cm.search_procedures("build", 10);
    assert_eq!(procs.len(), 1);
    assert_eq!(
        procs[0].steps,
        vec!["step-a".to_string(), "step-b".to_string()]
    );
}
