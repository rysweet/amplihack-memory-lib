//! Entity name extraction from text and concept fields.
//!
//! Extracts proper nouns (capitalized multi-word names) from text content.

use regex::Regex;
use std::sync::LazyLock;

/// Compiled regex matching multi-word proper names (e.g. "Sarah Chen", "O'Brien").
pub static MULTI_WORD_NAME_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"\b([A-Z][a-z]*(?:['\u{2019}\-][A-Z]?[a-z]+)+(?:\s+(?:[A-Z][a-z]+(?:['\u{2019}\-][A-Z]?[a-z]+)?))*|[A-Z][a-z]+(?:\s+(?:[A-Z][a-z]+(?:['\u{2019}\-][A-Z]?[a-z]+)?))+)\b"
    ).unwrap()
});

/// Extract the primary entity name from content or concept.
///
/// Uses simple heuristics to find proper nouns (capitalized multi-word names)
/// in the concept field first, then the content. Returns lowercase for
/// consistent indexing.
pub fn extract_entity_name(content: &str, concept: &str) -> Option<String> {
    for text in [concept, content] {
        if text.is_empty() {
            continue;
        }

        // Find capitalized multi-word names
        let matches: Vec<&str> = MULTI_WORD_NAME_RE
            .find_iter(text)
            .map(|m| m.as_str())
            .collect();

        if !matches.is_empty() {
            let best = matches.iter().max_by_key(|m| m.chars().count()).unwrap();
            return Some(best.to_lowercase());
        }

        // Single capitalized word that isn't at start of sentence
        let words: Vec<&str> = text.split_whitespace().collect();
        for (i, word) in words.iter().enumerate() {
            if i > 0 {
                let first_char = word.chars().next();
                if let Some(c) = first_char {
                    if c.is_uppercase() && word.chars().count() > 2 {
                        let cleaned: String = word
                            .trim_matches(|c: char| ".,;:!?()[]{}\"'".contains(c))
                            .to_string();
                        if !cleaned.is_empty()
                            && cleaned.chars().next().is_some_and(|c| c.is_uppercase())
                        {
                            return Some(cleaned.to_lowercase());
                        }
                    }
                }
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_word_name() {
        assert_eq!(
            extract_entity_name("Met with Sarah Chen today", ""),
            Some("sarah chen".to_string())
        );
    }

    #[test]
    fn test_concept_first() {
        assert_eq!(
            extract_entity_name("some content", "Sarah Chen"),
            Some("sarah chen".to_string())
        );
    }

    #[test]
    fn test_empty() {
        assert_eq!(extract_entity_name("", ""), None);
    }

    #[test]
    fn test_no_names() {
        assert_eq!(extract_entity_name("all lowercase words here", ""), None);
    }

    #[test]
    fn test_apostrophe_name() {
        let result = extract_entity_name("Met O'Brien today", "").unwrap();
        assert!(result.contains("o'brien") || result.contains("o\u{2019}brien"));
    }

    #[test]
    fn test_unicode_entity_extraction_cjk() {
        let result = extract_entity_name("hello 日本語 world", "");
        assert!(
            result.is_none(),
            "expected no entity from CJK text, got: {result:?}"
        );
    }

    #[test]
    fn test_unicode_entity_extraction_emoji() {
        let result = extract_entity_name("the 🎉🎊 Party was fun", "");
        assert_eq!(result, Some("party".to_string()));
    }
}
