use super::super::*;

// -- QueryValidator::is_safe_query --

#[test]
fn test_query_validator_safe() {
    assert!(QueryValidator::is_safe_query("SELECT * FROM experiences"));
    assert!(!QueryValidator::is_safe_query("DELETE FROM experiences"));
    assert!(!QueryValidator::is_safe_query("DROP TABLE experiences"));
}

#[test]
fn test_query_cost() {
    let cost = QueryValidator::estimate_cost("SELECT * FROM t");
    assert!(cost > 1); // Has full_scan penalty
}

#[test]
fn test_is_safe_query_with_created_at() {
    assert!(
        QueryValidator::is_safe_query("SELECT * FROM t WHERE created_at > '2024-01-01'"),
        "Column name 'created_at' should not trigger CREATE keyword match"
    );
}

#[test]
fn test_is_safe_query_with_updated_at() {
    assert!(
        QueryValidator::is_safe_query("SELECT * FROM t WHERE updated_at IS NOT NULL"),
        "Column name 'updated_at' should not trigger UPDATE keyword match"
    );
}

#[test]
fn test_is_safe_query_rejects_real_create() {
    assert!(!QueryValidator::is_safe_query(
        "SELECT 1; CREATE TABLE evil (id INT)"
    ));
}

#[test]
fn test_is_safe_query_insert_rejected() {
    assert!(!QueryValidator::is_safe_query(
        "INSERT INTO t VALUES (1, 'a')"
    ));
}

#[test]
fn test_is_safe_query_grant_rejected() {
    assert!(!QueryValidator::is_safe_query(
        "SELECT 1; GRANT ALL ON t TO user"
    ));
}

#[test]
fn test_is_safe_query_simple_select_accepted() {
    assert!(QueryValidator::is_safe_query(
        "SELECT id, name FROM users WHERE active = 1 LIMIT 100"
    ));
}

// -- Blocked keywords: sqlite_master, UNION, ATTACH, PRAGMA --

#[test]
fn test_is_safe_query_blocks_sqlite_master() {
    assert!(!QueryValidator::is_safe_query(
        "SELECT * FROM sqlite_master"
    ));
    assert!(!QueryValidator::is_safe_query(
        "SELECT * FROM sqlite_temp_master"
    ));
}

#[test]
fn test_is_safe_query_blocks_union() {
    assert!(!QueryValidator::is_safe_query("SELECT 1 UNION SELECT 2"));
}

#[test]
fn test_is_safe_query_blocks_attach() {
    assert!(!QueryValidator::is_safe_query(
        "SELECT 1; ATTACH DATABASE ':memory:' AS evil"
    ));
}

#[test]
fn test_is_safe_query_blocks_pragma() {
    assert!(!QueryValidator::is_safe_query(
        "SELECT 1; PRAGMA table_info(experiences)"
    ));
}

// -- QueryValidator::estimate_cost --

#[test]
fn test_estimate_cost_simple_with_limit() {
    let cost = QueryValidator::estimate_cost("SELECT id FROM t LIMIT 10");
    // Base(1) only — no full_scan, has LIMIT
    assert_eq!(cost, 1);
}

#[test]
fn test_estimate_cost_with_join() {
    let cost = QueryValidator::estimate_cost("SELECT * FROM a JOIN b ON a.id = b.id LIMIT 10");
    // Base(1) + full_scan(10) + join(5) = 16, has LIMIT so no +20
    assert!(cost >= 16, "Expected cost >= 16, got {cost}");
}

#[test]
fn test_estimate_cost_multiple_joins() {
    let cost = QueryValidator::estimate_cost(
        "SELECT * FROM a JOIN b ON a.id = b.id JOIN c ON b.id = c.id LIMIT 10",
    );
    // Base(1) + full_scan(10) + 2*join(5) = 21
    assert!(
        cost >= 21,
        "Expected cost >= 21 for double join, got {cost}"
    );
}

#[test]
fn test_estimate_cost_order_by() {
    let cost = QueryValidator::estimate_cost("SELECT * FROM t ORDER BY name LIMIT 10");
    // Base(1) + full_scan(10) + order_by(2) = 13
    assert!(cost >= 13, "Expected cost >= 13, got {cost}");
}

#[test]
fn test_estimate_cost_no_limit_penalty() {
    let cost_no_limit = QueryValidator::estimate_cost("SELECT * FROM t");
    let cost_with_limit = QueryValidator::estimate_cost("SELECT * FROM t LIMIT 10");
    assert!(
        cost_no_limit > cost_with_limit,
        "No-limit query should cost more"
    );
    // Difference should be the 20-point penalty
    assert_eq!(cost_no_limit - cost_with_limit, 20);
}

#[test]
fn test_estimate_cost_subquery() {
    let cost =
        QueryValidator::estimate_cost("SELECT * FROM t WHERE id IN (SELECT id FROM s) LIMIT 10");
    // Base(1) + full_scan(10) + subquery(3) = 14
    assert!(cost >= 14, "Expected cost >= 14 for subquery, got {cost}");
}

// -- QueryValidator::validate_query --

#[test]
fn test_validate_query_passes_under_limit() {
    assert!(QueryValidator::validate_query("SELECT id FROM t LIMIT 10", 50).is_ok());
}

#[test]
fn test_validate_query_fails_over_limit() {
    let result = QueryValidator::validate_query("SELECT * FROM t", 5);
    assert!(result.is_err());
    match result.unwrap_err() {
        MemoryError::QueryCostExceeded(msg) => {
            assert!(msg.contains("exceeds limit"));
        }
        other => panic!("expected QueryCostExceeded, got {other:?}"),
    }
}

#[test]
fn test_validate_query_at_exact_limit() {
    let cost = QueryValidator::estimate_cost("SELECT id FROM t LIMIT 10");
    // Passes when max_cost equals cost
    assert!(QueryValidator::validate_query("SELECT id FROM t LIMIT 10", cost).is_ok());
    // Fails when max_cost is one less
    assert!(QueryValidator::validate_query("SELECT id FROM t LIMIT 10", cost - 1).is_err());
}

// -- Cypher safety validation --

#[test]
fn test_is_safe_cypher_allows_match_return() {
    assert!(QueryValidator::is_safe_cypher(
        "MATCH (e:Experience) WHERE e.agent = $agent RETURN e.experience_id"
    ));
}

#[test]
fn test_is_safe_cypher_rejects_delete() {
    assert!(!QueryValidator::is_safe_cypher(
        "MATCH (e:Experience) DELETE e"
    ));
}

#[test]
fn test_is_safe_cypher_rejects_set() {
    assert!(!QueryValidator::is_safe_cypher(
        "MATCH (e:Experience) SET e.name = 'evil'"
    ));
}

#[test]
fn test_is_safe_cypher_rejects_create() {
    assert!(!QueryValidator::is_safe_cypher(
        "CREATE (:Experience {id: '1'})"
    ));
}

#[test]
fn test_is_safe_cypher_rejects_non_match() {
    assert!(!QueryValidator::is_safe_cypher(
        "DETACH DELETE (e:Experience)"
    ));
}

// -- QA audit --

#[test]
fn test_qa_rejects_semicolons() {
    assert!(!QueryValidator::is_safe_query("SELECT 1; DROP TABLE t"));
    assert!(!QueryValidator::is_safe_query("SELECT 1;"));
}

#[test]
fn test_qa_strips_line_comments() {
    assert!(QueryValidator::is_safe_query(
        "SELECT 1 -- comment\nFROM t LIMIT 1"
    ));
}

#[test]
fn test_qa_strips_block_comments() {
    assert!(!QueryValidator::is_safe_query(
        "SELECT /* x */ 1; DROP TABLE t"
    ));
}

#[test]
fn test_qa_rejects_vacuum() {
    assert!(!QueryValidator::is_safe_query("SELECT 1 FROM VACUUM"));
}

#[test]
fn test_qa_rejects_savepoint() {
    assert!(!QueryValidator::is_safe_query("SELECT 1 SAVEPOINT sp"));
}

#[test]
fn test_qa_rejects_begin_commit_rollback() {
    assert!(!QueryValidator::is_safe_query("SELECT 1 BEGIN"));
    assert!(!QueryValidator::is_safe_query("SELECT 1 COMMIT"));
    assert!(!QueryValidator::is_safe_query("SELECT 1 ROLLBACK"));
}

#[test]
fn test_qa_cypher_rejects_call() {
    assert!(!QueryValidator::is_safe_cypher(
        "MATCH (e:Experience) CALL db.info()"
    ));
}

#[test]
fn test_qa_cypher_rejects_load() {
    assert!(!QueryValidator::is_safe_cypher(
        "MATCH (e:Experience) LOAD CSV FROM '/etc/passwd'"
    ));
}

#[test]
fn test_qa_cypher_rejects_foreach() {
    assert!(!QueryValidator::is_safe_cypher(
        "MATCH (e:Experience) FOREACH (x IN [1] | SET e.v = x)"
    ));
}
