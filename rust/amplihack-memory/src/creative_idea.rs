//! Creative-idea prospective memory type + typed memory links.
//!
//! A *creative idea* is a first-class kind of [prospective
//! memory](crate::ProspectiveMemory): a future-oriented candidate
//! self-improvement that an agent generates, reviews, and either promotes to a
//! goal or parks. This module owns the **memory-model** parts of that type —
//! the lifecycle [`CreativeIdeaStatus`] state machine and the typed
//! [`MemoryLink`] edges to the other cognitive memory categories — so that
//! consumers (e.g. an agent runtime) orchestrate *around* the library rather
//! than re-deriving the lifecycle or the link taxonomy themselves.
//!
//! The type layers cleanly on top of the existing prospective primitive with
//! **no schema change**: a creative idea is a `ProspectiveMemory` whose
//! `trigger_condition` is the [`CREATIVE_IDEA_TRIGGER`] sentinel and whose
//! `action_on_trigger` carries a versioned JSON payload (see
//! [`CREATIVE_IDEA_PAYLOAD_VERSION`]). This module deliberately does **not**
//! own the review/metric payload shape — that is application orchestration —
//! but it *does* own the status vocabulary, the legal transitions, and the
//! link taxonomy, which are memory-architecture concerns.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

/// Prospective `trigger_condition` sentinel that marks a node as a creative
/// idea. This is the retrieval key for every stored creative-idea row, so it
/// is a **stable identifier** — renaming it is a breaking migration, not an
/// edit.
pub const CREATIVE_IDEA_TRIGGER: &str = "creative-idea";

/// On-disk payload schema version for a creative-idea prospective node.
///
/// A reader that encounters a row whose version is **newer** than this value
/// must fail closed rather than silently guess. Starts at `1`; a future
/// native-links migration bumps it.
pub const CREATIVE_IDEA_PAYLOAD_VERSION: u16 = 1;

/// The lifecycle status of a creative idea.
///
/// Status changes are **explicit and validated**: only the edges enumerated by
/// [`CreativeIdeaStatus::can_transition_to`] are legal, and callers are
/// expected to reject any other transition rather than overwrite the status.
///
/// # Examples
///
/// ```
/// use amplihack_memory::creative_idea::CreativeIdeaStatus;
///
/// assert!(CreativeIdeaStatus::New.can_transition_to(CreativeIdeaStatus::AcceptedForImplementation));
/// assert!(!CreativeIdeaStatus::Rejected.can_transition_to(CreativeIdeaStatus::New));
/// assert_eq!("NeedsHumanReview".parse(), Ok(CreativeIdeaStatus::NeedsHumanReview));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CreativeIdeaStatus {
    /// Freshly generated, not yet reviewed.
    New,
    /// A reviewer/synthesis step asked for a rewrite before acceptance.
    NeedsRevision,
    /// High-risk / flagged: a human must decide.
    NeedsHumanReview,
    /// Reviewed and accepted; may be promoted to a goal.
    AcceptedForImplementation,
    /// Terminal: rejected.
    Rejected,
    /// Parked; may be reconsidered later.
    Deferred,
    /// A goal/PR is in flight.
    ImplementationStarted,
    /// Terminal: completed — reachable only from [`Self::ImplementationStarted`].
    ImplementationCompleted,
}

impl CreativeIdeaStatus {
    /// Every status value, in a stable order.
    pub const ALL: [CreativeIdeaStatus; 8] = [
        CreativeIdeaStatus::New,
        CreativeIdeaStatus::NeedsRevision,
        CreativeIdeaStatus::NeedsHumanReview,
        CreativeIdeaStatus::AcceptedForImplementation,
        CreativeIdeaStatus::Rejected,
        CreativeIdeaStatus::Deferred,
        CreativeIdeaStatus::ImplementationStarted,
        CreativeIdeaStatus::ImplementationCompleted,
    ];

    /// Stable string form (matches the serde variant names).
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::New => "New",
            Self::NeedsRevision => "NeedsRevision",
            Self::NeedsHumanReview => "NeedsHumanReview",
            Self::AcceptedForImplementation => "AcceptedForImplementation",
            Self::Rejected => "Rejected",
            Self::Deferred => "Deferred",
            Self::ImplementationStarted => "ImplementationStarted",
            Self::ImplementationCompleted => "ImplementationCompleted",
        }
    }

    /// Terminal states have no outgoing transitions.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Rejected | Self::ImplementationCompleted)
    }

    /// Whether `self -> to` is an allowed edge of the lifecycle state machine.
    ///
    /// | From | To |
    /// |------|----|
    /// | `New` | `AcceptedForImplementation`, `Rejected`, `Deferred`, `NeedsRevision`, `NeedsHumanReview` |
    /// | `NeedsRevision` | `New`, `Rejected`, `Deferred` |
    /// | `NeedsHumanReview` | `AcceptedForImplementation`, `Rejected`, `Deferred` |
    /// | `Deferred` | `New`, `Rejected` |
    /// | `AcceptedForImplementation` | `ImplementationStarted`, `Deferred`, `Rejected` |
    /// | `ImplementationStarted` | `ImplementationCompleted`, `NeedsRevision`, `Rejected` |
    /// | `Rejected` / `ImplementationCompleted` | *(terminal)* |
    #[must_use]
    pub fn can_transition_to(&self, to: Self) -> bool {
        use CreativeIdeaStatus::{
            AcceptedForImplementation, Deferred, ImplementationCompleted, ImplementationStarted,
            NeedsHumanReview, NeedsRevision, New, Rejected,
        };
        matches!(
            (self, to),
            (New, AcceptedForImplementation)
                | (New, Rejected)
                | (New, Deferred)
                | (New, NeedsRevision)
                | (New, NeedsHumanReview)
                | (NeedsRevision, New)
                | (NeedsRevision, Rejected)
                | (NeedsRevision, Deferred)
                | (NeedsHumanReview, AcceptedForImplementation)
                | (NeedsHumanReview, Rejected)
                | (NeedsHumanReview, Deferred)
                | (Deferred, New)
                | (Deferred, Rejected)
                | (AcceptedForImplementation, ImplementationStarted)
                | (AcceptedForImplementation, Deferred)
                | (AcceptedForImplementation, Rejected)
                | (ImplementationStarted, ImplementationCompleted)
                | (ImplementationStarted, NeedsRevision)
                | (ImplementationStarted, Rejected)
        )
    }
}

impl fmt::Display for CreativeIdeaStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for CreativeIdeaStatus {
    type Err = ParseCreativeIdeaError;

    /// Parse a status string. **Fail-closed**: an unknown value is an error,
    /// never a silent default.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "New" => Ok(Self::New),
            "NeedsRevision" => Ok(Self::NeedsRevision),
            "NeedsHumanReview" => Ok(Self::NeedsHumanReview),
            "AcceptedForImplementation" => Ok(Self::AcceptedForImplementation),
            "Rejected" => Ok(Self::Rejected),
            "Deferred" => Ok(Self::Deferred),
            "ImplementationStarted" => Ok(Self::ImplementationStarted),
            "ImplementationCompleted" => Ok(Self::ImplementationCompleted),
            other => Err(ParseCreativeIdeaError {
                field: "status".to_string(),
                value: other.to_string(),
            }),
        }
    }
}

/// The cognitive memory category a [`MemoryLink`] points at.
///
/// A creative idea references the supporting resources it was built from: a
/// distilled fact ([`Self::Semantic`]), an autobiographical episode
/// ([`Self::Episodic`]), a reusable procedure ([`Self::Procedural`]), or a
/// standing goal ([`Self::Goal`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryLinkKind {
    /// A distilled semantic fact.
    Semantic,
    /// An autobiographical episode.
    Episodic,
    /// A reusable procedure.
    Procedural,
    /// A standing goal node.
    Goal,
}

impl MemoryLinkKind {
    /// Every link kind, in a stable order.
    pub const ALL: [MemoryLinkKind; 4] = [
        MemoryLinkKind::Semantic,
        MemoryLinkKind::Episodic,
        MemoryLinkKind::Procedural,
        MemoryLinkKind::Goal,
    ];

    /// Stable string form (matches the serde variant names).
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Semantic => "Semantic",
            Self::Episodic => "Episodic",
            Self::Procedural => "Procedural",
            Self::Goal => "Goal",
        }
    }
}

impl fmt::Display for MemoryLinkKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for MemoryLinkKind {
    type Err = ParseCreativeIdeaError;

    /// Parse a link-kind string. **Fail-closed** on an unknown value.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "Semantic" => Ok(Self::Semantic),
            "Episodic" => Ok(Self::Episodic),
            "Procedural" => Ok(Self::Procedural),
            "Goal" => Ok(Self::Goal),
            other => Err(ParseCreativeIdeaError {
                field: "link_kind".to_string(),
                value: other.to_string(),
            }),
        }
    }
}

/// A typed edge from a creative idea to another memory node that
/// supports/resources it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryLink {
    /// The category of node this link points at.
    pub kind: MemoryLinkKind,
    /// The `node_id` of the linked node.
    pub node_id: String,
}

impl MemoryLink {
    /// Construct a link of `kind` to `node_id`.
    #[must_use]
    pub fn new(kind: MemoryLinkKind, node_id: impl Into<String>) -> Self {
        Self {
            kind,
            node_id: node_id.into(),
        }
    }
}

/// Error returned when a persisted creative-idea enum string cannot be parsed.
///
/// Fail-closed: an unknown value is surfaced as this error rather than silently
/// coerced to a default.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseCreativeIdeaError {
    /// Which field failed to parse (e.g. `"status"`, `"link_kind"`).
    pub field: String,
    /// The offending value.
    pub value: String,
}

impl fmt::Display for ParseCreativeIdeaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "invalid creative-idea {}: unknown value '{}'",
            self.field, self.value
        )
    }
}

impl std::error::Error for ParseCreativeIdeaError {}

#[cfg(test)]
mod tests {
    use super::*;

    const ALLOWED: &[(CreativeIdeaStatus, CreativeIdeaStatus)] = &[
        (
            CreativeIdeaStatus::New,
            CreativeIdeaStatus::AcceptedForImplementation,
        ),
        (CreativeIdeaStatus::New, CreativeIdeaStatus::Rejected),
        (CreativeIdeaStatus::New, CreativeIdeaStatus::Deferred),
        (CreativeIdeaStatus::New, CreativeIdeaStatus::NeedsRevision),
        (
            CreativeIdeaStatus::New,
            CreativeIdeaStatus::NeedsHumanReview,
        ),
        (CreativeIdeaStatus::NeedsRevision, CreativeIdeaStatus::New),
        (
            CreativeIdeaStatus::NeedsRevision,
            CreativeIdeaStatus::Rejected,
        ),
        (
            CreativeIdeaStatus::NeedsRevision,
            CreativeIdeaStatus::Deferred,
        ),
        (
            CreativeIdeaStatus::NeedsHumanReview,
            CreativeIdeaStatus::AcceptedForImplementation,
        ),
        (
            CreativeIdeaStatus::NeedsHumanReview,
            CreativeIdeaStatus::Rejected,
        ),
        (
            CreativeIdeaStatus::NeedsHumanReview,
            CreativeIdeaStatus::Deferred,
        ),
        (CreativeIdeaStatus::Deferred, CreativeIdeaStatus::New),
        (CreativeIdeaStatus::Deferred, CreativeIdeaStatus::Rejected),
        (
            CreativeIdeaStatus::AcceptedForImplementation,
            CreativeIdeaStatus::ImplementationStarted,
        ),
        (
            CreativeIdeaStatus::AcceptedForImplementation,
            CreativeIdeaStatus::Deferred,
        ),
        (
            CreativeIdeaStatus::AcceptedForImplementation,
            CreativeIdeaStatus::Rejected,
        ),
        (
            CreativeIdeaStatus::ImplementationStarted,
            CreativeIdeaStatus::ImplementationCompleted,
        ),
        (
            CreativeIdeaStatus::ImplementationStarted,
            CreativeIdeaStatus::NeedsRevision,
        ),
        (
            CreativeIdeaStatus::ImplementationStarted,
            CreativeIdeaStatus::Rejected,
        ),
    ];

    #[test]
    fn status_transitions_match_table() {
        for &from in &CreativeIdeaStatus::ALL {
            for &to in &CreativeIdeaStatus::ALL {
                let expected = ALLOWED.contains(&(from, to));
                assert_eq!(
                    from.can_transition_to(to),
                    expected,
                    "transition {from} -> {to} mismatch"
                );
            }
        }
    }

    #[test]
    fn terminal_states_have_no_outgoing_edges() {
        for terminal in [
            CreativeIdeaStatus::Rejected,
            CreativeIdeaStatus::ImplementationCompleted,
        ] {
            assert!(terminal.is_terminal());
            for &to in &CreativeIdeaStatus::ALL {
                assert!(!terminal.can_transition_to(to));
            }
        }
    }

    #[test]
    fn completed_only_reachable_from_started() {
        for &from in &CreativeIdeaStatus::ALL {
            assert_eq!(
                from.can_transition_to(CreativeIdeaStatus::ImplementationCompleted),
                from == CreativeIdeaStatus::ImplementationStarted
            );
        }
    }

    #[test]
    fn status_string_roundtrips_and_fails_closed() {
        for &status in &CreativeIdeaStatus::ALL {
            assert_eq!(status.as_str().parse(), Ok(status));
        }
        let err = "bogus".parse::<CreativeIdeaStatus>().unwrap_err();
        assert_eq!(err.field, "status");
        assert_eq!(err.value, "bogus");
    }

    #[test]
    fn link_kind_string_roundtrips_and_fails_closed() {
        for &kind in &MemoryLinkKind::ALL {
            assert_eq!(kind.as_str().parse(), Ok(kind));
        }
        assert!("bogus".parse::<MemoryLinkKind>().is_err());
    }

    #[test]
    fn memory_link_serde_roundtrip() {
        let link = MemoryLink::new(MemoryLinkKind::Goal, "goal_42");
        let json = serde_json::to_string(&link).unwrap();
        let back: MemoryLink = serde_json::from_str(&json).unwrap();
        assert_eq!(link, back);
        assert!(json.contains("Goal"));
    }
}
