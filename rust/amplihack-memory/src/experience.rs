//! Experience data model for agent memories.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

/// Maximum number of characters allowed in the `context` field.
pub const MAX_CONTEXT_LENGTH: usize = 500;
/// Maximum number of characters allowed in the `outcome` field.
pub const MAX_OUTCOME_LENGTH: usize = 1000;

/// Types of experiences an agent can have.
///
/// Each variant carries different semantic weight when computing relevance
/// scores in [`crate::semantic_search::calculate_relevance`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ExperienceType {
    /// A task or action completed successfully.
    Success,
    /// A task or action that failed.
    Failure,
    /// A recurring pattern detected across multiple experiences.
    Pattern,
    /// A novel insight or learned lesson.
    Insight,
}

impl ExperienceType {
    /// Return the lowercase string representation of this experience type.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::Failure => "failure",
            Self::Pattern => "pattern",
            Self::Insight => "insight",
        }
    }
}

impl std::str::FromStr for ExperienceType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "success" => Ok(Self::Success),
            "failure" => Ok(Self::Failure),
            "pattern" => Ok(Self::Pattern),
            "insight" => Ok(Self::Insight),
            _ => Err(format!("unknown experience type: {s}")),
        }
    }
}

impl std::fmt::Display for ExperienceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Single agent experience record.
///
/// An experience captures a specific event in the agent's lifecycle together
/// with its context, outcome, and a confidence score. Experiences are
/// content-addressed: the `experience_id` is derived from the context,
/// outcome, and timestamp.
///
/// # Examples
///
/// ```
/// use amplihack_memory::{Experience, ExperienceType};
///
/// let exp = Experience::new(
///     ExperienceType::Success,
///     "compiled project".into(),
///     "zero warnings".into(),
///     0.95,
/// )
/// .unwrap();
/// assert!(exp.experience_id.starts_with("exp_"));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experience {
    /// Content-addressed unique identifier (e.g. `exp_20240101_120000_a1b2c3`).
    pub experience_id: String,
    /// Category of experience.
    pub experience_type: ExperienceType,
    /// Contextual description of what was happening (max 500 chars).
    pub context: String,
    /// What resulted from the experience (max 1000 chars).
    pub outcome: String,
    /// Confidence score in `[0.0, 1.0]`.
    pub confidence: f64,
    /// When the experience occurred.
    pub timestamp: DateTime<Utc>,
    /// Arbitrary key-value metadata.
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
    /// Categorical tags for filtering and similarity matching.
    #[serde(default)]
    pub tags: Vec<String>,
}

impl Experience {
    /// Create a new experience with validation.
    ///
    /// Generates a content-addressed `experience_id` from the context, outcome,
    /// and current timestamp.
    ///
    /// # Errors
    ///
    /// Returns [`crate::MemoryError::InvalidExperience`] if `context` is empty or
    /// exceeds 500 chars, `outcome` is empty or exceeds 1000 chars, or
    /// `confidence` is outside `[0.0, 1.0]`.
    pub fn new(
        experience_type: ExperienceType,
        context: String,
        outcome: String,
        confidence: f64,
    ) -> crate::Result<Self> {
        let timestamp = Utc::now();
        Self::with_timestamp(experience_type, context, outcome, confidence, timestamp)
    }

    /// Create a new experience with an explicit timestamp.
    ///
    /// Same validation rules as [`new`](Self::new).
    ///
    /// # Errors
    ///
    /// Returns [`crate::MemoryError::InvalidExperience`] on validation failure.
    pub fn with_timestamp(
        experience_type: ExperienceType,
        context: String,
        outcome: String,
        confidence: f64,
        timestamp: DateTime<Utc>,
    ) -> crate::Result<Self> {
        let experience_id = generate_id(&context, &outcome, &timestamp);

        let exp = Self {
            experience_id,
            experience_type,
            context,
            outcome,
            confidence,
            timestamp,
            metadata: HashMap::new(),
            tags: Vec::new(),
        };
        exp.validate()?;
        Ok(exp)
    }

    /// Create from a full set of fields (e.g. from a database row).
    ///
    /// Applies the same validation as [`with_timestamp`](Self::with_timestamp)
    /// but accepts a caller-supplied `experience_id` instead of generating one.
    ///
    /// # Errors
    ///
    /// Returns [`crate::MemoryError::InvalidExperience`] on validation failure.
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        experience_id: String,
        experience_type: ExperienceType,
        context: String,
        outcome: String,
        confidence: f64,
        timestamp: DateTime<Utc>,
        metadata: HashMap<String, serde_json::Value>,
        tags: Vec<String>,
    ) -> crate::Result<Self> {
        let exp = Self {
            experience_id,
            experience_type,
            context,
            outcome,
            confidence,
            timestamp,
            metadata,
            tags,
        };
        exp.validate()?;
        Ok(exp)
    }

    /// Validate all fields of this experience.
    ///
    /// # Errors
    ///
    /// Returns [`crate::MemoryError::InvalidExperience`] if `context` is empty or
    /// exceeds [`MAX_CONTEXT_LENGTH`] chars, `outcome` is empty or exceeds
    /// [`MAX_OUTCOME_LENGTH`] chars, or `confidence` is outside `[0.0, 1.0]`.
    pub fn validate(&self) -> crate::Result<()> {
        if self.context.trim().is_empty() {
            return Err(crate::MemoryError::InvalidExperience(
                "context cannot be empty".into(),
            ));
        }
        if self.context.chars().count() > MAX_CONTEXT_LENGTH {
            return Err(crate::MemoryError::InvalidExperience(format!(
                "context exceeds {MAX_CONTEXT_LENGTH} characters"
            )));
        }
        if self.outcome.trim().is_empty() {
            return Err(crate::MemoryError::InvalidExperience(
                "outcome cannot be empty".into(),
            ));
        }
        if self.outcome.chars().count() > MAX_OUTCOME_LENGTH {
            return Err(crate::MemoryError::InvalidExperience(format!(
                "outcome exceeds {MAX_OUTCOME_LENGTH} characters"
            )));
        }
        if !(0.0..=1.0).contains(&self.confidence) {
            return Err(crate::MemoryError::InvalidExperience(
                "confidence must be between 0.0 and 1.0".into(),
            ));
        }
        Ok(())
    }

    /// Serialize this experience into a flat key-value map suitable for JSON.
    pub fn to_map(&self) -> HashMap<String, serde_json::Value> {
        let mut map = HashMap::new();
        map.insert(
            "experience_id".into(),
            serde_json::Value::String(self.experience_id.clone()),
        );
        map.insert(
            "experience_type".into(),
            serde_json::Value::String(self.experience_type.as_str().into()),
        );
        map.insert(
            "context".into(),
            serde_json::Value::String(self.context.clone()),
        );
        map.insert(
            "outcome".into(),
            serde_json::Value::String(self.outcome.clone()),
        );
        map.insert("confidence".into(), serde_json::json!(self.confidence));
        map.insert(
            "timestamp".into(),
            serde_json::Value::String(self.timestamp.to_rfc3339()),
        );
        map.insert(
            "metadata".into(),
            serde_json::to_value(&self.metadata).unwrap_or_default(),
        );
        map.insert(
            "tags".into(),
            serde_json::to_value(&self.tags).unwrap_or_default(),
        );
        map
    }

    /// Deserialize from a key-value map (e.g. JSON round-trip).
    ///
    /// # Errors
    ///
    /// Returns [`crate::MemoryError::InvalidExperience`] if required fields are
    /// missing, unparseable, or fail validation.
    pub fn from_map(data: &HashMap<String, serde_json::Value>) -> crate::Result<Self> {
        let experience_id = data
            .get("experience_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| crate::MemoryError::InvalidExperience("missing experience_id".into()))?
            .to_string();

        let exp_type_str = data
            .get("experience_type")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                crate::MemoryError::InvalidExperience("missing experience_type".into())
            })?;
        let experience_type = exp_type_str.parse::<ExperienceType>().map_err(|_| {
            crate::MemoryError::InvalidExperience(format!(
                "unknown experience_type: {exp_type_str}"
            ))
        })?;

        let context = data
            .get("context")
            .and_then(|v| v.as_str())
            .ok_or_else(|| crate::MemoryError::InvalidExperience("missing context".into()))?
            .to_string();

        let outcome = data
            .get("outcome")
            .and_then(|v| v.as_str())
            .ok_or_else(|| crate::MemoryError::InvalidExperience("missing outcome".into()))?
            .to_string();

        let confidence = data
            .get("confidence")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| crate::MemoryError::InvalidExperience("missing confidence".into()))?;

        let timestamp_str = data
            .get("timestamp")
            .and_then(|v| v.as_str())
            .ok_or_else(|| crate::MemoryError::InvalidExperience("missing timestamp".into()))?;
        let timestamp = DateTime::parse_from_rfc3339(timestamp_str)
            .map(|dt| dt.with_timezone(&Utc))
            .map_err(|e| {
                crate::MemoryError::InvalidExperience(format!("invalid timestamp: {e}"))
            })?;

        let metadata: HashMap<String, serde_json::Value> = data
            .get("metadata")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();

        let tags: Vec<String> = data
            .get("tags")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();

        Self::from_parts(
            experience_id,
            experience_type,
            context,
            outcome,
            confidence,
            timestamp,
            metadata,
            tags,
        )
    }
}

impl PartialEq for Experience {
    fn eq(&self, other: &Self) -> bool {
        self.experience_id == other.experience_id
    }
}

impl Eq for Experience {}

impl std::hash::Hash for Experience {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.experience_id.hash(state);
    }
}

fn generate_id(context: &str, outcome: &str, timestamp: &DateTime<Utc>) -> String {
    let date_str = timestamp.format("%Y%m%d").to_string();
    let time_str = timestamp.format("%H%M%S").to_string();
    let content = format!("{context}{outcome}{}", timestamp.to_rfc3339());
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    let hash = hasher.finalize();
    let hash_str = &crate::utils::hex_encode(hash)[..16];
    format!("exp_{date_str}_{time_str}_{hash_str}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_experience() {
        let exp = Experience::new(
            ExperienceType::Success,
            "test context".into(),
            "test outcome".into(),
            0.9,
        )
        .unwrap();
        assert!(exp.experience_id.starts_with("exp_"));
        assert_eq!(exp.experience_type, ExperienceType::Success);
        assert_eq!(exp.confidence, 0.9);
    }

    #[test]
    fn test_empty_context_rejected() {
        let result = Experience::new(ExperienceType::Success, "".into(), "outcome".into(), 0.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_confidence_out_of_range() {
        let result = Experience::new(ExperienceType::Success, "ctx".into(), "outcome".into(), 1.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_experience_type_roundtrip() {
        for et in [
            ExperienceType::Success,
            ExperienceType::Failure,
            ExperienceType::Pattern,
            ExperienceType::Insight,
        ] {
            assert_eq!(et.as_str().parse::<ExperienceType>(), Ok(et));
        }
    }

    #[test]
    fn test_experience_equality() {
        let e1 =
            Experience::new(ExperienceType::Success, "ctx".into(), "outcome".into(), 0.5).unwrap();
        let mut e2 = e1.clone();
        assert_eq!(e1, e2);
        e2.experience_id = "different".into();
        assert_ne!(e1, e2);
    }

    #[test]
    fn test_to_from_map() {
        let exp = Experience::new(
            ExperienceType::Insight,
            "some context".into(),
            "some outcome".into(),
            0.75,
        )
        .unwrap();
        let map = exp.to_map();
        let restored = Experience::from_map(&map).unwrap();
        assert_eq!(exp.experience_id, restored.experience_id);
        assert_eq!(exp.experience_type, restored.experience_type);
        assert_eq!(exp.context, restored.context);
    }

    #[test]
    fn test_from_parts_valid() {
        let result = Experience::from_parts(
            "test_id".into(),
            ExperienceType::Success,
            "valid context".into(),
            "valid outcome".into(),
            0.8,
            Utc::now(),
            Default::default(),
            vec!["tag1".into()],
        );
        assert!(result.is_ok());
        let exp = result.unwrap();
        assert_eq!(exp.experience_id, "test_id");
        assert_eq!(exp.confidence, 0.8);
    }

    #[test]
    fn test_from_parts_empty_context() {
        let result = Experience::from_parts(
            "id".into(),
            ExperienceType::Success,
            "".into(),
            "outcome".into(),
            0.5,
            Utc::now(),
            Default::default(),
            vec![],
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_from_parts_context_too_long() {
        let result = Experience::from_parts(
            "id".into(),
            ExperienceType::Success,
            "x".repeat(501),
            "outcome".into(),
            0.5,
            Utc::now(),
            Default::default(),
            vec![],
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_from_parts_empty_outcome() {
        let result = Experience::from_parts(
            "id".into(),
            ExperienceType::Success,
            "context".into(),
            "   ".into(),
            0.5,
            Utc::now(),
            Default::default(),
            vec![],
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_from_parts_outcome_too_long() {
        let result = Experience::from_parts(
            "id".into(),
            ExperienceType::Success,
            "context".into(),
            "x".repeat(1001),
            0.5,
            Utc::now(),
            Default::default(),
            vec![],
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_from_parts_confidence_out_of_range() {
        let result = Experience::from_parts(
            "id".into(),
            ExperienceType::Success,
            "context".into(),
            "outcome".into(),
            1.5,
            Utc::now(),
            Default::default(),
            vec![],
        );
        assert!(result.is_err());

        let result = Experience::from_parts(
            "id".into(),
            ExperienceType::Success,
            "context".into(),
            "outcome".into(),
            -0.1,
            Utc::now(),
            Default::default(),
            vec![],
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_experience_id_hash_length() {
        let exp = Experience::new(
            ExperienceType::Success,
            "test context".into(),
            "test outcome".into(),
            0.9,
        )
        .unwrap();
        // ID format: exp_YYYYMMDD_HHMMSS_<hash>
        let parts: Vec<&str> = exp.experience_id.split('_').collect();
        assert_eq!(
            parts.len(),
            4,
            "ID should have 4 underscore-separated parts"
        );
        let hash_part = parts[3];
        assert_eq!(
            hash_part.len(),
            16,
            "Hash suffix should be 16 hex chars, got {}: {}",
            hash_part.len(),
            hash_part
        );
        assert!(
            hash_part.chars().all(|c| c.is_ascii_hexdigit()),
            "Hash suffix should be hex"
        );
    }

    #[test]
    fn test_experience_id_uniqueness() {
        let exp1 = Experience::new(
            ExperienceType::Success,
            "context alpha".into(),
            "outcome one".into(),
            0.8,
        )
        .unwrap();
        let exp2 = Experience::new(
            ExperienceType::Success,
            "context beta".into(),
            "outcome two".into(),
            0.8,
        )
        .unwrap();
        assert_ne!(
            exp1.experience_id, exp2.experience_id,
            "Different inputs should produce different IDs"
        );
    }
}
