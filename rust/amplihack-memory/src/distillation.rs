//! Deterministic episodic→semantic distillation heuristics.
//!
//! The storage side of the episodic→semantic transition already lives in
//! [`CognitiveMemory`](crate::CognitiveMemory) (`store_episode`,
//! `mark_episode_distilled`, `list_undistilled_episodes`, `store_fact`,
//! `upsert_fact` dedup). What the crate lacked was a deterministic *extraction*
//! heuristic that turns a free-text episode into the atomic semantic-fact
//! candidates worth persisting. Without one, a consumer distils an episode as a
//! single opaque blob — roughly one fact per episode — and under-yields durable
//! facts from compound episodes such as `"A happened; B happened."`.
//!
//! This module fills that gap with a pure, allocation-only, fully deterministic
//! distiller:
//!
//! * [`distill_episode`] segments one episode into atomic facts by sentence and
//!   conjunctive-clause boundaries, normalises whitespace/punctuation, and drops
//!   trivial fragments.
//! * [`distill_corpus`] runs [`distill_episode`] over many episodes and
//!   exact-dedups the results (case-insensitively, insertion-order preserved) so
//!   repeated episodes do not inflate the yield.
//!
//! # Fact-yield benchmark
//!
//! [`FACT_YIELD_BENCH_CORPUS`] is a fixed episodic corpus and
//! [`fact_yield`] counts the durable semantic facts distilled from it. The
//! prior, naive "one fact per episode" behaviour ([`fact_yield_baseline`])
//! yields [`FACT_YIELD_BASELINE`] facts; the atomic-segmentation distiller in
//! this module yields [`FACT_YIELD_IMPROVED`]. The gap is the measurable
//! improvement, and the regression test in this module asserts the improved
//! yield never drops back to the baseline.
//!
//! ## Determinism
//!
//! Output depends only on the input string: no clocks, randomness, or
//! hash-ordered iteration leaks into the returned order. Calling any function
//! here twice on the same input always returns identical results.
//!
//! ## Known limitations
//!
//! Sentence splitting treats `.`, `!`, `?` followed by whitespace or
//! end-of-text as a boundary, so an abbreviation like `"e.g. "` splits early;
//! the resulting fragment is filtered by the [`MIN_FACT_WORDS`] noise gate. This
//! keeps the heuristic simple and dependency-free; richer segmentation can be
//! layered on later without changing the public contract.

use std::collections::HashSet;

/// Minimum number of whitespace-separated words a segment must contain to be
/// retained as a durable fact. Filters trivial fragments like `"ok"` / `"done"`.
pub const MIN_FACT_WORDS: usize = 3;

/// Minimum number of words required on *each* side of a conjunctive boundary
/// before a clause is split in two. Prevents shredding short list items
/// (`"nuts and bolts"`) into meaningless fragments.
pub const MIN_CLAUSE_WORDS: usize = 3;

/// Conjunctive separators that split a sentence into independent clauses.
///
/// A boundary only becomes a split when both neighbours are substantive
/// (`>= MIN_CLAUSE_WORDS` words); otherwise the separator is preserved and the
/// parts are re-joined.
const CLAUSE_SEPARATORS: &[&str] = &["; ", ", and ", ", but ", ", so "];

/// A fixed episodic corpus used by the deterministic fact-yield benchmark.
///
/// It deliberately mixes single-clause episodes, multi-sentence episodes,
/// conjunctive compound sentences, and one exact-duplicate episode (indices 0
/// and 5) so the benchmark exercises segmentation and dedup together.
pub const FACT_YIELD_BENCH_CORPUS: &[&str] = &[
    "Sarah Chen joined the platform team. She leads the caching project.",
    "The deploy failed because the migration timed out; the rollback succeeded.",
    "We adopted Rust for the memory library, and we dropped the Python backend.",
    "Kuzu stores facts as nodes. Kuzu stores relationships as edges.",
    "The incident was caused by a config typo.",
    // Exact duplicate of the first episode — must not inflate the yield.
    "Sarah Chen joined the platform team. She leads the caching project.",
    "Retrieval uses Graph RAG. No ML embeddings are required.",
    "The team agreed to weekly reviews, and the team agreed to pair on migrations.",
];

/// Durable fact-yield of the naive "one fact per episode" baseline on
/// [`FACT_YIELD_BENCH_CORPUS`] (the pre-improvement behaviour this module
/// replaces). Recorded so the regression test can prove a measurable gain.
pub const FACT_YIELD_BASELINE: usize = 7;

/// Durable fact-yield of the atomic-segmentation distiller in this module on
/// [`FACT_YIELD_BENCH_CORPUS`]. Strictly greater than [`FACT_YIELD_BASELINE`].
pub const FACT_YIELD_IMPROVED: usize = 13;

/// Distil a single episode's free text into atomic semantic-fact candidates.
///
/// Splits on sentence terminators (`.`, `!`, `?`) then on conjunctive clause
/// boundaries, normalises each fragment (collapsing internal whitespace and
/// stripping surrounding punctuation), and discards fragments shorter than
/// [`MIN_FACT_WORDS`] words. The returned order follows the text.
///
/// ```
/// use amplihack_memory::distillation::distill_episode;
///
/// let facts = distill_episode("Ada designed the engine; Ada wrote the notes.");
/// assert_eq!(
///     facts,
///     vec![
///         "Ada designed the engine".to_string(),
///         "Ada wrote the notes".to_string(),
///     ],
/// );
/// ```
pub fn distill_episode(content: &str) -> Vec<String> {
    let mut facts = Vec::new();
    for sentence in split_sentences(content) {
        for clause in split_clauses(&sentence) {
            if let Some(fact) = normalize_fact(&clause) {
                facts.push(fact);
            }
        }
    }
    facts
}

/// Distil a corpus of episodes into de-duplicated durable facts.
///
/// Each episode is distilled with [`distill_episode`]; the results are
/// concatenated and exact-deduplicated using a case-insensitive key, preserving
/// first-seen order. Repeated episodes therefore contribute their facts only
/// once.
///
/// ```
/// use amplihack_memory::distillation::distill_corpus;
///
/// let facts = distill_corpus(&[
///     "The cache was warmed. The cache was warmed.",
///     "The cache was warmed.",
/// ]);
/// assert_eq!(facts, vec!["The cache was warmed".to_string()]);
/// ```
pub fn distill_corpus<S: AsRef<str>>(episodes: &[S]) -> Vec<String> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut durable = Vec::new();
    for episode in episodes {
        for fact in distill_episode(episode.as_ref()) {
            if seen.insert(fact.to_lowercase()) {
                durable.push(fact);
            }
        }
    }
    durable
}

/// Count the durable semantic facts the improved distiller extracts from
/// `episodes` — the fact-yield metric.
pub fn fact_yield<S: AsRef<str>>(episodes: &[S]) -> usize {
    distill_corpus(episodes).len()
}

/// Count the durable semantic facts the naive baseline extracts from
/// `episodes`, treating each whole episode as a single fact (after the same
/// normalisation and exact-dedup). This models the pre-improvement behaviour
/// and backs [`FACT_YIELD_BASELINE`].
pub fn fact_yield_baseline<S: AsRef<str>>(episodes: &[S]) -> usize {
    let mut seen: HashSet<String> = HashSet::new();
    let mut count = 0;
    for episode in episodes {
        if let Some(fact) = normalize_fact(episode.as_ref()) {
            if seen.insert(fact.to_lowercase()) {
                count += 1;
            }
        }
    }
    count
}

// ---------------------------------------------------------------------------
// internal segmentation helpers
// ---------------------------------------------------------------------------

fn is_terminator(c: char) -> bool {
    c == '.' || c == '!' || c == '?'
}

fn word_count(text: &str) -> usize {
    text.split_whitespace().count()
}

/// Split text into sentences on `.`/`!`/`?` boundaries.
///
/// A terminator ends a sentence only when it is followed by whitespace or the
/// end of the text, so intra-token dots (`"3.14"`) do not split. Runs of
/// terminators (`"?!"`, `"..."`) collapse into a single boundary.
fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();
    let mut chars = text.chars().peekable();

    while let Some(c) = chars.next() {
        current.push(c);
        if !is_terminator(c) {
            continue;
        }
        // Absorb any adjacent run of terminators.
        while let Some(&next) = chars.peek() {
            if is_terminator(next) {
                chars.next();
            } else {
                break;
            }
        }
        let boundary = match chars.peek() {
            None => true,
            Some(&next) => next.is_whitespace(),
        };
        if boundary {
            sentences.push(std::mem::take(&mut current));
        }
    }

    if !current.trim().is_empty() {
        sentences.push(current);
    }
    sentences
}

/// Split a sentence into independent clauses on the [`CLAUSE_SEPARATORS`],
/// keeping only splits where both sides are substantive.
fn split_clauses(sentence: &str) -> Vec<String> {
    let mut clauses = vec![sentence.to_string()];
    for sep in CLAUSE_SEPARATORS {
        clauses = clauses
            .into_iter()
            .flat_map(|clause| split_on_separator(&clause, sep))
            .collect();
    }
    clauses
}

/// Split `clause` on every occurrence of `sep`, but only accept a boundary when
/// the accumulated left side and the next part each have at least
/// [`MIN_CLAUSE_WORDS`] words; otherwise the separator is preserved.
fn split_on_separator(clause: &str, sep: &str) -> Vec<String> {
    let parts: Vec<&str> = clause.split(sep).collect();
    if parts.len() < 2 {
        return vec![clause.to_string()];
    }

    let mut out: Vec<String> = Vec::new();
    let mut buf = parts[0].to_string();
    for part in &parts[1..] {
        if word_count(&buf) >= MIN_CLAUSE_WORDS && word_count(part) >= MIN_CLAUSE_WORDS {
            out.push(std::mem::take(&mut buf));
            buf = (*part).to_string();
        } else {
            buf.push_str(sep);
            buf.push_str(part);
        }
    }
    out.push(buf);
    out
}

/// Normalise a raw segment into a fact, or return `None` if it is too trivial.
///
/// Collapses internal whitespace to single spaces and strips surrounding
/// whitespace and terminal punctuation; internal punctuation is preserved.
/// Returns `None` when fewer than [`MIN_FACT_WORDS`] words remain.
fn normalize_fact(segment: &str) -> Option<String> {
    let collapsed = segment.split_whitespace().collect::<Vec<_>>().join(" ");
    let trimmed = collapsed
        .trim_matches(|c: char| c.is_whitespace() || ".,;:!?".contains(c))
        .to_string();
    if word_count(&trimmed) < MIN_FACT_WORDS {
        return None;
    }
    Some(trimmed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_compound_sentence_into_atomic_facts() {
        let facts =
            distill_episode("Sarah Chen joined the platform team. She leads the caching project.");
        assert_eq!(
            facts,
            vec![
                "Sarah Chen joined the platform team".to_string(),
                "She leads the caching project".to_string(),
            ]
        );
    }

    #[test]
    fn splits_semicolon_and_conjunction_clauses() {
        assert_eq!(
            distill_episode(
                "The deploy failed because the migration timed out; the rollback succeeded."
            ),
            vec![
                "The deploy failed because the migration timed out".to_string(),
                "the rollback succeeded".to_string(),
            ]
        );
        assert_eq!(
            distill_episode(
                "We adopted Rust for the memory library, and we dropped the Python backend."
            ),
            vec![
                "We adopted Rust for the memory library".to_string(),
                "we dropped the Python backend".to_string(),
            ]
        );
    }

    #[test]
    fn does_not_over_split_short_list_items() {
        // "nuts and bolts" is not comma-and separated and each side of a bare
        // conjunction is too short, so the clause stays intact.
        assert_eq!(
            distill_episode("The kit ships with nuts and bolts."),
            vec!["The kit ships with nuts and bolts".to_string()]
        );
    }

    #[test]
    fn keeps_intra_token_dots_intact() {
        assert_eq!(
            distill_episode("The value of pi is 3.14 here."),
            vec!["The value of pi is 3.14 here".to_string()]
        );
    }

    #[test]
    fn drops_trivial_fragments() {
        assert!(distill_episode("ok. done. yes.").is_empty());
    }

    #[test]
    fn corpus_dedups_repeated_episodes() {
        let facts = distill_corpus(&["The cache was warmed.", "The cache was warmed."]);
        assert_eq!(facts, vec!["The cache was warmed".to_string()]);
    }

    #[test]
    fn distillation_is_deterministic() {
        let first = distill_corpus(FACT_YIELD_BENCH_CORPUS);
        let second = distill_corpus(FACT_YIELD_BENCH_CORPUS);
        assert_eq!(first, second, "distillation must be order-stable");
    }

    #[test]
    fn baseline_yield_matches_recorded_constant() {
        assert_eq!(
            fact_yield_baseline(FACT_YIELD_BENCH_CORPUS),
            FACT_YIELD_BASELINE,
            "recorded baseline must match the naive one-fact-per-episode yield",
        );
    }

    #[test]
    fn improved_yield_matches_recorded_constant() {
        assert_eq!(
            fact_yield(FACT_YIELD_BENCH_CORPUS),
            FACT_YIELD_IMPROVED,
            "improved yield drifted from the recorded benchmark value",
        );
        // Every distilled fact is distinct (no duplicates leaked into the count).
        let facts = distill_corpus(FACT_YIELD_BENCH_CORPUS);
        let unique: HashSet<String> = facts.iter().map(|f| f.to_lowercase()).collect();
        assert_eq!(unique.len(), facts.len(), "durable facts must be distinct");
    }

    /// Regression guard: the improved distiller must never silently regress to
    /// (or below) the recorded baseline yield, and must stay a measurable gain.
    #[test]
    fn fact_yield_never_regresses_below_baseline() {
        let improved = fact_yield(FACT_YIELD_BENCH_CORPUS);
        let baseline = fact_yield_baseline(FACT_YIELD_BENCH_CORPUS);

        assert!(
            improved >= FACT_YIELD_BASELINE,
            "fact-yield {improved} regressed below baseline {FACT_YIELD_BASELINE}",
        );
        assert!(
            improved >= FACT_YIELD_IMPROVED,
            "fact-yield {improved} regressed below recorded improved yield {FACT_YIELD_IMPROVED}",
        );
        assert!(
            improved > baseline,
            "improved yield {improved} must exceed baseline {baseline}",
        );
    }
}
