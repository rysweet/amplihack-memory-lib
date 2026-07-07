//! Shared utility functions.

/// Hex-encode a byte slice to a lowercase hex string.
pub(crate) fn hex_encode(bytes: impl AsRef<[u8]>) -> String {
    bytes.as_ref().iter().map(|b| format!("{b:02x}")).collect()
}

/// Is `name` a safe SQL/Cypher identifier — `^[A-Za-z_][A-Za-z0-9_]*$`?
///
/// Shared by the feature-agnostic graph layer (e.g. validating the `order_by`
/// column of [`GraphStore::query_nodes_ordered`](crate::graph::GraphStore::query_nodes_ordered)
/// before it is spliced into an `ORDER BY` clause). The durable backends keep
/// their own identically-defined validators for `node_type` / filter-key
/// checks; this mirrors that rule so every layer rejects the same inputs.
pub(crate) fn is_valid_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_valid_identifier_accepts_plain_names() {
        assert!(is_valid_identifier("priority"));
        assert!(is_valid_identifier("_underscore"));
        assert!(is_valid_identifier("col1"));
        assert!(is_valid_identifier("Trigger_Condition"));
    }

    #[test]
    fn test_is_valid_identifier_rejects_injection() {
        assert!(!is_valid_identifier(""));
        assert!(!is_valid_identifier("1col"));
        assert!(!is_valid_identifier("a b"));
        assert!(!is_valid_identifier("x) DELETE n //"));
        assert!(!is_valid_identifier("1;DROP"));
    }
}
