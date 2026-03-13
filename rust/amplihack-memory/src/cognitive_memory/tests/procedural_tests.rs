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
