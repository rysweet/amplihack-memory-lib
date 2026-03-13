use std::collections::HashMap;

/// Extract normalized pattern key from discovery.
pub fn extract_pattern_key(discovery: &HashMap<String, serde_json::Value>) -> String {
    let disc_type = discovery
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");

    if let Some(link_type) = discovery.get("link_type").and_then(|v| v.as_str()) {
        return format!("{disc_type}_{link_type}");
    }

    if let Some(file) = discovery.get("file").and_then(|v| v.as_str()) {
        if file.ends_with(".md") {
            let prefix = if file.contains('_') {
                file.split('_').next().unwrap_or(file)
            } else {
                file.rsplit('.').next_back().unwrap_or(file)
            };
            return format!("{prefix}_{disc_type}");
        }
        return disc_type.to_string();
    }

    disc_type.to_string()
}

/// Calculate pattern confidence based on occurrences.
pub fn calculate_pattern_confidence(occurrences: usize) -> f64 {
    (0.5 + (occurrences as f64 * 0.1)).min(0.95)
}
