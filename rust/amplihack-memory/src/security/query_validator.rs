//! SQL query validation and cost estimation.

use regex::Regex;
use std::sync::LazyLock;

use crate::errors::MemoryError;

/// Validate and estimate cost of database queries.
///
/// Assigns cost points based on SQL patterns (JOINs, subqueries,
/// unbounded SELECTs) and rejects queries exceeding a configured limit.
pub struct QueryValidator;

static COST_PATTERNS: LazyLock<Vec<(&str, Regex, i32)>> = LazyLock::new(|| {
    vec![
        (
            "full_scan",
            Regex::new(r"(?i)SELECT\s+\*\s+FROM\s+\w+\s*(?:WHERE)?").unwrap(),
            10,
        ),
        ("join", Regex::new(r"(?i)\bJOIN\b").unwrap(), 5),
        (
            "subquery",
            Regex::new(r"(?is)SELECT.*\(.*SELECT").unwrap(),
            3,
        ),
        ("order_by", Regex::new(r"(?i)\bORDER BY\b").unwrap(), 2),
        // no_limit is checked separately since Rust regex doesn't support lookahead
    ]
});

static LIMIT_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)\bLIMIT\b").unwrap());

impl QueryValidator {
    /// Estimate the cost of a SQL query based on pattern analysis.
    ///
    /// Cost factors: full table scans (+10), JOINs (+5 each), subqueries (+3),
    /// ORDER BY (+2), and missing LIMIT (+20). Base cost is 1.
    pub fn estimate_cost(sql: &str) -> i32 {
        let mut cost = 1; // Base cost

        for (name, pattern, points) in COST_PATTERNS.iter() {
            if *name == "join" {
                let matches = pattern.find_iter(sql).count();
                cost += (matches as i32) * points;
            } else if pattern.is_match(sql) {
                cost += points;
            }
        }

        // Check for no LIMIT (penalize unbounded queries)
        let sql_upper = sql.to_uppercase();
        if sql_upper.contains("SELECT") && !LIMIT_RE.is_match(sql) {
            cost += 20;
        }

        cost
    }

    /// Validate that a query's estimated cost does not exceed `max_cost`.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::QueryCostExceeded`] if the cost exceeds the limit.
    pub fn validate_query(sql: &str, max_cost: i32) -> crate::Result<()> {
        let cost = Self::estimate_cost(sql);
        if cost > max_cost {
            return Err(MemoryError::QueryCostExceeded(format!(
                "Query cost ({cost}) exceeds limit ({max_cost}). \
                 Consider adding LIMIT, reducing JOINs, or simplifying query."
            )));
        }
        Ok(())
    }

    /// Check if a Cypher query is non-destructive (read-only).
    ///
    /// Allows `MATCH ... RETURN` queries and rejects destructive operations
    /// like `DELETE`, `SET`, `REMOVE`, `MERGE`, and non-schema `CREATE`.
    pub fn is_safe_cypher(cypher: &str) -> bool {
        let trimmed = cypher.trim();
        if !trimmed
            .chars()
            .take(5)
            .collect::<String>()
            .eq_ignore_ascii_case("match")
        {
            return false;
        }

        static DESTRUCTIVE_CYPHER: LazyLock<Vec<Regex>> = LazyLock::new(|| {
            ["DELETE", "DETACH", "SET", "REMOVE", "MERGE", "CREATE"]
                .iter()
                .map(|kw| Regex::new(&format!(r"(?i)\b{kw}\b")).unwrap())
                .collect()
        });

        !DESTRUCTIVE_CYPHER.iter().any(|p| p.is_match(trimmed))
    }

    /// Check if a SQL query is non-destructive (read-only SELECT).
    ///
    /// Returns `false` for DELETE, UPDATE, INSERT, DROP, and other DDL/DML.
    pub fn is_safe_query(sql: &str) -> bool {
        let sql_trimmed = sql.trim();

        if !sql_trimmed
            .chars()
            .take(6)
            .collect::<String>()
            .eq_ignore_ascii_case("select")
        {
            return false;
        }

        static DESTRUCTIVE_PATTERNS: LazyLock<Vec<Regex>> = LazyLock::new(|| {
            [
                "DELETE", "UPDATE", "INSERT", "DROP", "TRUNCATE", "ALTER", "CREATE", "REPLACE",
                "GRANT", "REVOKE", "UNION", "ATTACH", "PRAGMA",
            ]
            .iter()
            .map(|kw| Regex::new(&format!(r"(?i)\b{kw}\b")).unwrap())
            .collect()
        });

        // Block access to sqlite_master / sqlite_temp_master metadata tables.
        static METADATA_TABLE_RE: LazyLock<Regex> =
            LazyLock::new(|| Regex::new(r"(?i)\bsqlite_(?:temp_)?master\b").unwrap());

        if METADATA_TABLE_RE.is_match(sql_trimmed) {
            return false;
        }

        for pattern in DESTRUCTIVE_PATTERNS.iter() {
            if pattern.is_match(sql_trimmed) {
                return false;
            }
        }

        true
    }
}
