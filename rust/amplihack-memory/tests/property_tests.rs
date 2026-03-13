use proptest::prelude::*;
use std::collections::HashMap;

use amplihack_memory::*;

// ---------------------------------------------------------------------------
// 1. Experience roundtrip: arbitrary Experience -> validate -> fields preserved
// ---------------------------------------------------------------------------
proptest! {
    #[test]
    fn experience_roundtrip(
        context in "[a-zA-Z0-9]{1}[a-zA-Z0-9 ]{0,199}",
        outcome in "[a-zA-Z0-9]{1}[a-zA-Z0-9 ]{0,199}",
        confidence in 0.0f64..=1.0,
    ) {
        let exp = Experience::new(
            ExperienceType::Success,
            context.clone(),
            outcome.clone(),
            confidence,
        ).unwrap();
        prop_assert_eq!(&exp.context, &context);
        prop_assert_eq!(&exp.outcome, &outcome);
        prop_assert!((exp.confidence - confidence).abs() < f64::EPSILON);
    }
}

// ---------------------------------------------------------------------------
// 2. Similarity is symmetric: sim(a,b) == sim(b,a)
// ---------------------------------------------------------------------------
proptest! {
    #[test]
    fn similarity_symmetric(
        a in "[a-z ]{1,100}",
        b in "[a-z ]{1,100}",
    ) {
        let sim_ab = compute_word_similarity(&a, &b);
        let sim_ba = compute_word_similarity(&b, &a);
        prop_assert!((sim_ab - sim_ba).abs() < f64::EPSILON,
            "sim({:?},{:?})={} != sim({:?},{:?})={}",
            a, b, sim_ab, b, a, sim_ba);
    }
}

// ---------------------------------------------------------------------------
// 3. Similarity self-identity: sim(a,a) == 1.0 when tokens exist
// ---------------------------------------------------------------------------
proptest! {
    #[test]
    fn similarity_self_identity(
        // Ensure at least one token >=3 chars by including a fixed word
        word in "[a-z]{3,15}",
        padding in "[a-z]{0,30}",
    ) {
        let text = format!("{word} {padding}");
        let sim = compute_word_similarity(&text, &text);
        prop_assert!((sim - 1.0).abs() < f64::EPSILON,
            "self-similarity of {:?} = {} (expected 1.0)", text, sim);
    }
}

// ---------------------------------------------------------------------------
// 4. Similarity is bounded [0.0, 1.0]
// ---------------------------------------------------------------------------
proptest! {
    #[test]
    fn similarity_bounded(
        a in ".*",
        b in ".*",
    ) {
        let sim = compute_word_similarity(&a, &b);
        prop_assert!((0.0..=1.0).contains(&sim),
            "sim({:?},{:?})={} out of [0,1]", a, b, sim);
    }
}

// ---------------------------------------------------------------------------
// 5. Tag similarity is symmetric
// ---------------------------------------------------------------------------
proptest! {
    #[test]
    fn tag_similarity_symmetric(
        tags_a in prop::collection::vec("[a-z]{2,10}", 0..8),
        tags_b in prop::collection::vec("[a-z]{2,10}", 0..8),
    ) {
        let sim_ab = compute_tag_similarity(&tags_a, &tags_b);
        let sim_ba = compute_tag_similarity(&tags_b, &tags_a);
        prop_assert!((sim_ab - sim_ba).abs() < f64::EPSILON,
            "tag sim asymmetric: {} vs {}", sim_ab, sim_ba);
    }
}

// ---------------------------------------------------------------------------
// 6. Entity extraction idempotency: extract(text) applied twice yields same result
// ---------------------------------------------------------------------------
proptest! {
    #[test]
    fn entity_extraction_idempotent(
        content in "[A-Z][a-z]{2,10}( [A-Z][a-z]{2,10}){0,3}",
        concept in "[a-z ]{3,30}",
    ) {
        let first = extract_entity_name(&content, &concept);
        if let Some(ref name) = first {
            let second = extract_entity_name(name, &concept);
            // Re-extracting from an already-extracted name should either
            // return the same value or None (no proper nouns left).
            if let Some(ref name2) = second {
                prop_assert_eq!(name, name2,
                    "entity extraction not idempotent: {:?} -> {:?}", name, name2);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 7. Contradiction detection is symmetric
// ---------------------------------------------------------------------------
proptest! {
    #[test]
    fn contradiction_symmetric(
        concept in "[a-z]{3,12}",
        num_a in 1u32..1000,
        num_b in 1u32..1000,
    ) {
        let content_a = format!("The {concept} has {num_a} items");
        let content_b = format!("The {concept} has {num_b} items");

        let result_ab = detect_contradiction(&content_a, &content_b, &concept, &concept);
        let result_ba = detect_contradiction(&content_b, &content_a, &concept, &concept);

        match (&result_ab, &result_ba) {
            (Some(ab), Some(ba)) => {
                prop_assert_eq!(ab.contradiction, ba.contradiction,
                    "contradiction asymmetric for {} vs {}", num_a, num_b);
            }
            (None, None) => {} // both agree no contradiction
            _ => prop_assert!(false,
                "one direction found contradiction, other didn't: {:?} vs {:?}",
                result_ab, result_ba),
        }
    }
}

// ---------------------------------------------------------------------------
// 8. Experience validation rejects invalid confidence
// ---------------------------------------------------------------------------
proptest! {
    #[test]
    fn invalid_confidence_rejected(
        over in 1.001f64..100.0,
        under in -100.0f64..-0.001,
    ) {
        let result_over = Experience::new(
            ExperienceType::Success,
            "valid context".into(),
            "valid outcome".into(),
            over,
        );
        prop_assert!(result_over.is_err(),
            "confidence {} should be rejected", over);

        let result_under = Experience::new(
            ExperienceType::Success,
            "valid context".into(),
            "valid outcome".into(),
            under,
        );
        prop_assert!(result_under.is_err(),
            "confidence {} should be rejected", under);
    }
}

// ---------------------------------------------------------------------------
// 9. Experience serialization roundtrip via to_map / from_map
// ---------------------------------------------------------------------------
proptest! {
    #[test]
    fn experience_serde_roundtrip(
        context in "[a-zA-Z0-9]{1}[a-zA-Z0-9 ]{0,99}",
        outcome in "[a-zA-Z0-9]{1}[a-zA-Z0-9 ]{0,99}",
        confidence in 0.0f64..=1.0,
    ) {
        let exp = Experience::new(
            ExperienceType::Pattern,
            context.clone(),
            outcome.clone(),
            confidence,
        ).unwrap();
        let map = exp.to_map();
        let restored = Experience::from_map(&map).unwrap();
        prop_assert_eq!(&restored.context, &context);
        prop_assert_eq!(&restored.outcome, &outcome);
        prop_assert_eq!(&restored.experience_id, &exp.experience_id);
        prop_assert!((restored.confidence - confidence).abs() < f64::EPSILON);
    }
}

// ---------------------------------------------------------------------------
// 10. Composite similarity is bounded [0.0, 1.0]
// ---------------------------------------------------------------------------
proptest! {
    #[test]
    fn composite_similarity_bounded(
        content_a in "[a-z ]{0,50}",
        content_b in "[a-z ]{0,50}",
        tags_a in prop::collection::vec("[a-z]{2,8}", 0..5),
        tags_b in prop::collection::vec("[a-z]{2,8}", 0..5),
    ) {
        let node_a = build_node(&content_a, &tags_a, "");
        let node_b = build_node(&content_b, &tags_b, "");
        let sim = compute_similarity(&node_a, &node_b);
        prop_assert!((0.0..=1.0).contains(&sim),
            "composite similarity {} out of [0,1]", sim);
    }
}

fn build_node(content: &str, tags: &[String], concept: &str) -> HashMap<String, serde_json::Value> {
    let mut m = HashMap::new();
    m.insert("content".into(), serde_json::Value::String(content.into()));
    m.insert(
        "tags".into(),
        serde_json::Value::Array(
            tags.iter()
                .map(|t| serde_json::Value::String(t.clone()))
                .collect(),
        ),
    );
    m.insert("concept".into(), serde_json::Value::String(concept.into()));
    m
}
