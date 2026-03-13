//! Search implementation for the Ladybug backend.

use ladybug_graph_rs::Property;

use super::LadybugBackend;
use crate::experience::{Experience, ExperienceType};
use crate::security::QueryValidator;

/// Execute a text search against experiences in the Ladybug graph.
///
/// Matches experiences whose context or outcome contain `query` (case-insensitive).
pub(crate) fn search_experiences(
    backend: &LadybugBackend,
    query: &str,
    experience_type: Option<ExperienceType>,
    min_confidence: f64,
    limit: usize,
) -> crate::Result<Vec<Experience>> {
    let mut where_clauses = vec![
        "e.agent_name = $agent".to_string(),
        "e.confidence >= $min_conf".to_string(),
        "(lower(e.context) CONTAINS lower($query) OR \
          lower(e.outcome) CONTAINS lower($query))"
            .to_string(),
    ];
    let mut params: Vec<(&str, Property)> = vec![
        ("agent", Property::from(backend.agent_name.as_str())),
        ("min_conf", Property::Double(min_confidence)),
        ("query", Property::from(query)),
    ];

    let type_str;
    if let Some(et) = experience_type {
        where_clauses.push("e.experience_type = $exp_type".to_string());
        type_str = et.as_str().to_string();
        params.push(("exp_type", Property::from(type_str.as_str())));
    }

    let where_clause = where_clauses.join(" AND ");
    // Safety: `limit` is a `usize`, guaranteed to be a non-negative integer.
    let cypher = format!(
        "MATCH (e:Experience) WHERE {where_clause} \
         RETURN {} \
         ORDER BY e.timestamp DESC \
         LIMIT {limit}",
        LadybugBackend::RETURN_COLS
    );

    if !QueryValidator::is_safe_cypher(&cypher) {
        return Err(crate::errors::MemoryError::SecurityViolation(
            "constructed Cypher query failed safety check".into(),
        ));
    }

    let rows = backend.exec(&cypher, &params)?;
    rows.iter()
        .map(|r| LadybugBackend::row_to_experience(r))
        .collect()
}
