use super::super::*;
use crate::memory_types::MemoryCategory;

fn make_mem() -> HierarchicalMemory {
    HierarchicalMemory::new("test_agent").unwrap()
}

// -- Memory classification --

#[test]
fn test_classifier_procedural() {
    let c = MemoryClassifier::new();
    assert_eq!(
        c.classify("How to bake a cake", ""),
        MemoryCategory::Procedural
    );
    assert_eq!(
        c.classify("Step 1: preheat oven", ""),
        MemoryCategory::Procedural
    );
}

#[test]
fn test_classifier_prospective() {
    let c = MemoryClassifier::new();
    assert_eq!(
        c.classify("I plan to visit Paris", ""),
        MemoryCategory::Prospective
    );
    assert_eq!(
        c.classify("My future goal is to learn Rust", ""),
        MemoryCategory::Prospective
    );
}

#[test]
fn test_classifier_episodic() {
    let c = MemoryClassifier::new();
    assert_eq!(
        c.classify("Something interesting happened today", ""),
        MemoryCategory::Episodic
    );
    assert_eq!(
        c.classify("I observed a rare bird", ""),
        MemoryCategory::Episodic
    );
}

#[test]
fn test_classifier_semantic_default() {
    let c = MemoryClassifier::new();
    assert_eq!(
        c.classify("Water is composed of hydrogen and oxygen", ""),
        MemoryCategory::Semantic
    );
}

#[test]
fn test_auto_classification_on_store() {
    let mut mem = make_mem();
    let nid = mem
        .store_knowledge(
            "Step 1: install Rust. Step 2: write code.",
            "tutorial",
            0.9,
            None,
            "",
            &[],
            None,
        )
        .unwrap();

    let nodes = mem.get_all_knowledge(None, 100);
    let stored = nodes.iter().find(|n| n.node_id == nid).unwrap();
    assert_eq!(stored.category, MemoryCategory::Procedural);
}

// -- Extended coverage: MemoryClassifier::classify --

#[test]
fn test_classifier_concept_triggers_category() {
    let c = MemoryClassifier::new();
    // Concept alone contains the keyword
    assert_eq!(
        c.classify("generic statement", "future plans"),
        MemoryCategory::Prospective
    );
}

#[test]
fn test_classifier_priority_procedural_over_episodic() {
    let c = MemoryClassifier::new();
    // Contains both "step" (procedural) and "happened" (episodic)
    // Procedural is checked first
    assert_eq!(
        c.classify("step by step what happened", ""),
        MemoryCategory::Procedural
    );
}

#[test]
fn test_classifier_case_insensitive() {
    let c = MemoryClassifier::new();
    assert_eq!(
        c.classify("HOW TO BUILD A HOUSE", ""),
        MemoryCategory::Procedural
    );
    assert_eq!(
        c.classify("I SAW AN EVENT THAT OCCURRED", ""),
        MemoryCategory::Episodic
    );
}

#[test]
fn test_classifier_default_returns() {
    let c = MemoryClassifier;
    assert_eq!(c.classify("plain fact", ""), MemoryCategory::Semantic);
}
