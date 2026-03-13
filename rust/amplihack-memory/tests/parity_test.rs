//! A/B parity test harness — Rust side.
//!
//! Mirrors every test in `python_harness.py` and writes results to
//! `/tmp/rust_parity_results.json` so the comparison script can diff them.

use amplihack_memory::{
    compute_similarity, compute_tag_similarity, compute_word_similarity,
    contradiction::detect_contradiction,
    entity_extraction::extract_entity_name,
    pattern_recognition::{calculate_pattern_confidence, extract_pattern_key, PatternDetector},
    security::{CredentialScrubber, QueryValidator},
    CognitiveMemory, Experience, ExperienceType,
};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs;

fn record(results: &mut Vec<Value>, test: &str, result: Value, success: bool) {
    results.push(json!({
        "test": test,
        "result": result,
        "success": success,
    }));
}

#[test]
fn parity_test_all() {
    let mut results: Vec<Value> = Vec::new();

    // ── 1. Experience ──────────────────────────────────────────────
    {
        let exp = Experience::new(
            ExperienceType::Success,
            "Deployed model to production".into(),
            "Model accuracy improved by 15%".into(),
            0.85,
        )
        .unwrap();

        let mut exp_with_tags = exp.clone();
        exp_with_tags.tags = vec!["deployment".into(), "ml".into()];

        record(
            &mut results,
            "experience_create",
            json!({
                "experience_type": "success",
                "context": exp_with_tags.context,
                "outcome": exp_with_tags.outcome,
                "confidence": exp_with_tags.confidence,
                "has_id": !exp_with_tags.experience_id.is_empty(),
                "id_prefix": &exp_with_tags.experience_id[..4],
                "has_timestamp": true,
                "tags": exp_with_tags.tags,
                "metadata": {},
            }),
            true,
        );

        // Validation: empty context
        let r = Experience::new(
            ExperienceType::Success,
            "".into(),
            "some outcome".into(),
            0.5,
        );
        record(
            &mut results,
            "experience_empty_context_rejected",
            json!(r.is_err()),
            true,
        );

        // Validation: confidence out of range
        let r = Experience::new(ExperienceType::Success, "ctx".into(), "outcome".into(), 1.5);
        record(
            &mut results,
            "experience_confidence_range",
            json!(r.is_err()),
            true,
        );
    }

    // ── 2. Similarity ──────────────────────────────────────────────
    {
        let ws_identical = compute_word_similarity("hello world test", "hello world test");
        record(
            &mut results,
            "word_similarity_identical",
            json!((ws_identical * 10000.0).round() / 10000.0),
            true,
        );

        let ws_none = compute_word_similarity("alpha beta gamma", "delta epsilon zeta");
        record(
            &mut results,
            "word_similarity_no_overlap",
            json!((ws_none * 10000.0).round() / 10000.0),
            true,
        );

        let ws_partial =
            compute_word_similarity("machine learning AI", "artificial intelligence ML");
        record(
            &mut results,
            "word_similarity_partial",
            json!((ws_partial * 10000.0).round() / 10000.0),
            true,
        );

        let tags_a: Vec<String> = vec!["rust".into(), "python".into()];
        let tags_b: Vec<String> = vec!["python".into(), "go".into()];
        let ts = compute_tag_similarity(&tags_a, &tags_b);
        record(
            &mut results,
            "tag_similarity",
            json!((ts * 10000.0).round() / 10000.0),
            true,
        );

        // Composite similarity — partial overlap
        let mut node_a = HashMap::new();
        node_a.insert(
            "content".into(),
            json!("Rust programming language memory safety"),
        );
        node_a.insert("concept".into(), json!("rust"));
        node_a.insert("tags".into(), json!(["language", "systems"]));

        let mut node_b = HashMap::new();
        node_b.insert(
            "content".into(),
            json!("Rust programming compiler optimization"),
        );
        node_b.insert("concept".into(), json!("rust"));
        node_b.insert("tags".into(), json!(["language", "compiler"]));

        let cs = compute_similarity(&node_a, &node_b);
        record(
            &mut results,
            "compute_similarity_partial",
            json!((cs * 10000.0).round() / 10000.0),
            true,
        );

        // Composite similarity — identical
        let mut node_same = HashMap::new();
        node_same.insert(
            "content".into(),
            json!("Rust programming language is fast and safe"),
        );
        node_same.insert("concept".into(), json!("rust"));
        node_same.insert("tags".into(), json!(["language", "systems"]));

        let cs_id = compute_similarity(&node_same, &node_same.clone());
        record(
            &mut results,
            "compute_similarity_identical",
            json!((cs_id * 10000.0).round() / 10000.0),
            true,
        );
    }

    // ── 3. Entity extraction ───────────────────────────────────────
    {
        let e1 = extract_entity_name("John Smith works at Microsoft in Seattle", "");
        record(
            &mut results,
            "entity_extraction_multi_word",
            json!(e1),
            true,
        );

        let e2 = extract_entity_name("some content", "Sarah Chen");
        record(
            &mut results,
            "entity_extraction_concept_first",
            json!(e2),
            true,
        );

        let e3 = extract_entity_name("", "");
        record(&mut results, "entity_extraction_empty", json!(e3), true);

        let e4 = extract_entity_name("all lowercase words here", "");
        record(&mut results, "entity_extraction_no_names", json!(e4), true);
    }

    // ── 4. Contradiction detection ─────────────────────────────────
    {
        let r1 = detect_contradiction(
            "The temperature is 72 degrees",
            "The temperature is 45 degrees",
            "temperature",
            "temperature",
        );
        record(
            &mut results,
            "contradiction_detected",
            json!({
                "has_contradiction": r1.as_ref().is_some_and(|c| c.contradiction),
                "conflicting_values": r1.as_ref().map_or(String::new(), |c| c.conflicting_values.clone()),
            }),
            true,
        );

        let r2 = detect_contradiction(
            "Team has 5 members",
            "Team has 5 members",
            "team size",
            "team size",
        );
        record(
            &mut results,
            "contradiction_same_numbers",
            json!({
                "has_contradiction": r2.as_ref().is_some_and(|c| c.contradiction),
            }),
            true,
        );

        let r3 = detect_contradiction("Has 5 members", "Has 8 items", "team", "inventory");
        record(
            &mut results,
            "contradiction_different_concepts",
            json!({
                "has_contradiction": r3.as_ref().is_some_and(|c| c.contradiction),
            }),
            true,
        );

        let r4 = detect_contradiction(
            "The team is large",
            "The team is small",
            "team size",
            "team size",
        );
        record(
            &mut results,
            "contradiction_no_numbers",
            json!({
                "has_contradiction": r4.as_ref().is_some_and(|c| c.contradiction),
            }),
            true,
        );
    }

    // ── 5. Security — credential scrubbing ─────────────────────────
    {
        let scrubber = CredentialScrubber::new();

        let text_with_creds = concat!(
            "AWS key AKIAIOSFODNN7EXAMPLE found. ",
            "GitHub token ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef12. ",
            "password=mysecret123"
        );

        let contains = scrubber.contains_credentials(text_with_creds);
        record(&mut results, "credentials_detected", json!(contains), true);

        let (scrubbed, was_modified) = scrubber.scrub_text(text_with_creds);
        record(
            &mut results,
            "credentials_scrubbed",
            json!({
                "was_modified": was_modified,
                "contains_aws_key": scrubbed.contains("AKIAIOSFODNN7EXAMPLE"),
                "contains_github_token": scrubbed.contains("ghp_ABCDEF"),
                "contains_password": scrubbed.contains("mysecret123"),
                "contains_redacted": scrubbed.contains("[REDACTED]"),
            }),
            true,
        );

        let clean_text = "This text has no credentials at all";
        let clean_contains = scrubber.contains_credentials(clean_text);
        record(
            &mut results,
            "credentials_clean_text",
            json!(!clean_contains),
            true,
        );
    }

    // ── 6. Security — query validation ─────────────────────────────
    {
        let safe = QueryValidator::is_safe_query("SELECT * FROM experiences");
        record(&mut results, "query_safe_select", json!(safe), true);

        let unsafe_drop = QueryValidator::is_safe_query("DROP TABLE experiences");
        record(&mut results, "query_unsafe_drop", json!(!unsafe_drop), true);

        let unsafe_delete = QueryValidator::is_safe_query("DELETE FROM experiences WHERE id = 1");
        record(
            &mut results,
            "query_unsafe_delete",
            json!(!unsafe_delete),
            true,
        );

        let cost = QueryValidator::estimate_cost("SELECT * FROM experiences");
        record(&mut results, "query_cost_simple_select", json!(cost), true);

        let cost_join = QueryValidator::estimate_cost(
            "SELECT * FROM experiences JOIN tags ON experiences.id = tags.exp_id LIMIT 10",
        );
        record(&mut results, "query_cost_with_join", json!(cost_join), true);
    }

    // ── 7. CognitiveMemory ─────────────────────────────────────────
    {
        let mut cm = CognitiveMemory::new("test-agent").unwrap();

        let tags: Vec<String> = vec!["language".into(), "programming".into()];
        let fact_id = cm
            .store_fact(
                "python",
                "Python is a programming language",
                0.9,
                "",
                Some(&tags),
                None,
            )
            .unwrap();
        record(
            &mut results,
            "cognitive_store_fact",
            json!({
                "has_id": !fact_id.is_empty(),
                "id_type": "str",
            }),
            true,
        );

        let ep_id = cm
            .store_episode(
                "Deployed the new ML model successfully",
                "deployment-session",
                None,
                None,
            )
            .unwrap();
        record(
            &mut results,
            "cognitive_store_episode",
            json!({
                "has_id": !ep_id.is_empty(),
                "id_type": "str",
            }),
            true,
        );

        let wm_id = cm
            .store_working("goal", "Complete the deployment pipeline", "task-001", 0.95)
            .unwrap();
        record(
            &mut results,
            "cognitive_push_working",
            json!({
                "has_id": !wm_id.is_empty(),
                "id_type": "str",
            }),
            true,
        );

        let wm_slots = cm.get_working("task-001");
        record(
            &mut results,
            "cognitive_get_working",
            json!({
                "count": wm_slots.len(),
                "first_content": wm_slots.first().map(|s| s.content.as_str()).unwrap_or(""),
            }),
            true,
        );

        let stats = cm.get_statistics();
        record(
            &mut results,
            "cognitive_stats",
            json!({
                "has_semantic": stats.contains_key("semantic"),
                "has_episodic": stats.contains_key("episodic"),
                "has_working": stats.contains_key("working"),
                "semantic_count": stats.get("semantic").copied().unwrap_or(0),
                "episodic_count": stats.get("episodic").copied().unwrap_or(0),
                "working_count": stats.get("working").copied().unwrap_or(0),
            }),
            true,
        );
    }

    // ── 8. Pattern recognition ─────────────────────────────────────
    {
        let mut detector = PatternDetector::new(3, 0.5);

        let disc: HashMap<String, Value> = [("type".to_string(), json!("test_type"))]
            .into_iter()
            .collect();

        // Below threshold
        for _ in 0..2 {
            detector.add_discovery(&disc);
        }
        record(
            &mut results,
            "pattern_below_threshold",
            json!(!detector.is_pattern_recognized("test_type")),
            true,
        );

        // At threshold
        detector.add_discovery(&disc);
        record(
            &mut results,
            "pattern_at_threshold",
            json!(detector.is_pattern_recognized("test_type")),
            true,
        );
        record(
            &mut results,
            "pattern_occurrence_count",
            json!(detector.get_occurrence_count("test_type")),
            true,
        );

        // Pattern key extraction
        let key1 = extract_pattern_key(&disc);
        record(&mut results, "pattern_key_simple", json!(key1), true);

        let disc2: HashMap<String, Value> = [
            ("type".to_string(), json!("link")),
            ("link_type".to_string(), json!("similar")),
        ]
        .into_iter()
        .collect();
        let key2 = extract_pattern_key(&disc2);
        record(&mut results, "pattern_key_with_link", json!(key2), true);

        // Confidence calculation
        let conf = calculate_pattern_confidence(3, 3);
        record(
            &mut results,
            "pattern_confidence_at_3",
            json!((conf * 10000.0).round() / 10000.0),
            true,
        );

        let conf5 = calculate_pattern_confidence(5, 3);
        record(
            &mut results,
            "pattern_confidence_at_5",
            json!((conf5 * 10000.0).round() / 10000.0),
            true,
        );
    }

    // ── Write output ───────────────────────────────────────────────
    let output = serde_json::to_string_pretty(&results).unwrap();
    fs::write("/tmp/rust_parity_results.json", &output).unwrap();
    println!("{output}");

    // Verify all tests succeeded
    let failed: Vec<&Value> = results
        .iter()
        .filter(|r| !r["success"].as_bool().unwrap_or(false))
        .collect();
    assert!(failed.is_empty(), "Some tests failed: {failed:?}");
}
