use std::collections::HashMap;

use crate::experience::Experience;

use super::PatternDetector;

/// Recognize new patterns from discoveries.
pub fn recognize_patterns(
    current_discoveries: &[HashMap<String, serde_json::Value>],
    known_patterns: Option<&[Experience]>,
    threshold: usize,
) -> Vec<Experience> {
    let mut detector = PatternDetector::new(threshold, 0.5);

    for discovery in current_discoveries {
        detector.add_discovery(discovery);
    }

    let mut new_patterns = detector.get_recognized_patterns(None);

    if let Some(known) = known_patterns {
        let known_keys: std::collections::HashSet<String> = known
            .iter()
            .filter_map(|p| {
                if let Some(start) = p.context.find("Pattern '") {
                    let start = start + "Pattern '".len();
                    if let Some(rest) = p.context.get(start..) {
                        if let Some(end) = rest.find('\'') {
                            return Some(p.context[start..start + end].to_string());
                        }
                    }
                }
                if p.context.to_lowercase().contains("known_pattern") {
                    return Some("known_pattern".to_string());
                }
                None
            })
            .collect();

        new_patterns.retain(|p| {
            if let Some(start) = p.context.find("Pattern '") {
                let start = start + "Pattern '".len();
                if let Some(rest) = p.context.get(start..) {
                    if let Some(end) = rest.find('\'') {
                        let key = &p.context[start..start + end];
                        return !known_keys.contains(key);
                    }
                }
            }
            true
        });
    }

    new_patterns
}
