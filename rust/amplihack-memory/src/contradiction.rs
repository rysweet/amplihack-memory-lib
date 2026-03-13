//! Contradiction detection between knowledge facts.
//!
//! Detects when two facts about the same concept contain conflicting numerical
//! values, indicating an update or contradiction.

use regex::Regex;
use std::collections::{BTreeSet, HashSet};
use std::sync::LazyLock;

static NUMBER_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\b\d+(?:\.\d+)?\b").unwrap());

/// Result of contradiction detection between two facts.
#[derive(Debug, Clone, PartialEq)]
pub struct ContradictionResult {
    /// `true` if the two facts contain conflicting numerical values.
    pub contradiction: bool,
    /// Human-readable description of the conflicting values (e.g. `"5 vs 8"`).
    pub conflicting_values: String,
}

/// Detect if two facts about the same concept contain contradictory numbers.
pub fn detect_contradiction(
    content_a: &str,
    content_b: &str,
    concept_a: &str,
    concept_b: &str,
) -> Option<ContradictionResult> {
    let concept_words_a: HashSet<String> = if concept_a.is_empty() {
        HashSet::new()
    } else {
        concept_a
            .to_lowercase()
            .split_whitespace()
            .map(String::from)
            .collect()
    };

    let concept_words_b: HashSet<String> = if concept_b.is_empty() {
        HashSet::new()
    } else {
        concept_b
            .to_lowercase()
            .split_whitespace()
            .map(String::from)
            .collect()
    };

    if concept_words_a.is_empty() || concept_words_b.is_empty() {
        return None;
    }

    let common: HashSet<&String> = concept_words_a
        .intersection(&concept_words_b)
        .filter(|w| w.len() > 2)
        .collect();

    if common.is_empty() {
        return None;
    }

    let nums_a: BTreeSet<String> = NUMBER_RE
        .find_iter(content_a)
        .map(|m| m.as_str().to_string())
        .collect();
    let nums_b: BTreeSet<String> = NUMBER_RE
        .find_iter(content_b)
        .map(|m| m.as_str().to_string())
        .collect();

    if nums_a.is_empty() || nums_b.is_empty() {
        return None;
    }

    let unique_to_a: BTreeSet<&String> = nums_a.difference(&nums_b).collect();
    let unique_to_b: BTreeSet<&String> = nums_b.difference(&nums_a).collect();

    if !unique_to_a.is_empty() && !unique_to_b.is_empty() {
        let a_vals: Vec<String> = unique_to_a.iter().map(|s| s.to_string()).collect();
        let b_vals: Vec<String> = unique_to_b.iter().map(|s| s.to_string()).collect();
        return Some(ContradictionResult {
            contradiction: true,
            conflicting_values: format!("{} vs {}", a_vals.join(", "), b_vals.join(", ")),
        });
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contradiction_detected() {
        let result = detect_contradiction(
            "The team has 5 members",
            "The team has 8 members",
            "team size",
            "team size",
        );
        assert!(result.is_some());
        let r = result.unwrap();
        assert!(r.contradiction);
    }

    #[test]
    fn test_no_contradiction_different_concepts() {
        let result = detect_contradiction("Has 5 members", "Has 8 items", "team", "inventory");
        assert!(result.is_none());
    }

    #[test]
    fn test_no_contradiction_same_numbers() {
        let result = detect_contradiction(
            "Team has 5 members",
            "Team has 5 members",
            "team size",
            "team size",
        );
        assert!(result.is_none());
    }

    #[test]
    fn test_no_numbers() {
        let result = detect_contradiction(
            "The team is large",
            "The team is small",
            "team size",
            "team size",
        );
        assert!(result.is_none());
    }
}
