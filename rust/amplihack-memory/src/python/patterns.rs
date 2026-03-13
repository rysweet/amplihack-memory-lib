//! PyO3 wrapper for the `PatternDetector` struct.
#![allow(clippy::useless_conversion)]

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::pattern_recognition::PatternDetector;

use super::experience::PyExperience;
use super::helpers::py_to_json;

/// Pattern detector that identifies recurring patterns from discoveries.
#[pyclass(name = "PatternDetector")]
pub struct PyPatternDetector {
    inner: PatternDetector,
}

#[pymethods]
impl PyPatternDetector {
    /// Create a new PatternDetector.
    ///
    /// Args:
    ///     threshold: Number of observations before promoting a pattern
    ///     min_confidence: Minimum confidence (0.0–1.0) for recognized patterns
    #[new]
    #[pyo3(signature = (threshold=3, min_confidence=0.5))]
    fn new(threshold: usize, min_confidence: f64) -> Self {
        Self {
            inner: PatternDetector::new(threshold, min_confidence),
        }
    }

    /// Record a discovery for pattern tracking.
    ///
    /// Args:
    ///     discovery: Dict describing the discovery (must include "type" key)
    fn add_discovery(&mut self, discovery: &Bound<'_, PyDict>) -> PyResult<()> {
        let map = dict_to_hashmap(discovery)?;
        self.inner.add_discovery(&map);
        Ok(())
    }

    /// Record an outcome to validate a pattern.
    ///
    /// Args:
    ///     pattern_type: The pattern key to validate
    ///     success: Whether the outcome was successful
    fn record_outcome(&mut self, pattern_type: &str, success: bool) {
        self.inner.validate_pattern(pattern_type, success);
    }

    /// Get occurrence count for a pattern type.
    fn get_occurrence_count(&self, pattern_type: &str) -> usize {
        self.inner.get_occurrence_count(pattern_type)
    }

    /// Check if a pattern has been recognized (met threshold).
    fn is_pattern_recognized(&self, pattern_type: &str) -> bool {
        self.inner.is_pattern_recognized(pattern_type)
    }

    /// Detect recognized patterns and return them as Experience objects.
    #[pyo3(signature = (min_confidence=None))]
    fn detect_patterns(&self, min_confidence: Option<f64>) -> Vec<PyExperience> {
        self.inner
            .get_recognized_patterns(min_confidence)
            .into_iter()
            .map(PyExperience::from_rust)
            .collect()
    }

    /// Get all recognized patterns as a list of dicts.
    #[pyo3(signature = (min_confidence=None))]
    fn get_all_patterns(&self, py: Python<'_>, min_confidence: Option<f64>) -> PyResult<PyObject> {
        let patterns = self.inner.get_recognized_patterns(min_confidence);
        let list: Vec<PyObject> = patterns
            .iter()
            .map(|exp| {
                let d = PyDict::new_bound(py);
                d.set_item("id", &exp.experience_id)?;
                d.set_item("context", &exp.context)?;
                d.set_item("outcome", &exp.outcome)?;
                d.set_item("confidence", exp.confidence)?;
                d.set_item("timestamp", exp.timestamp.to_rfc3339())?;
                Ok(d.to_object(py))
            })
            .collect::<PyResult<_>>()?;
        Ok(PyList::new_bound(py, list).to_object(py))
    }

    /// Get pattern confidence for a specific pattern type.
    ///
    /// Returns the confidence score if the pattern exists, or None.
    fn get_pattern_confidence(&self, pattern_type: &str) -> Option<f64> {
        if self.inner.is_pattern_recognized(pattern_type) {
            // Detect patterns and find the matching one
            let patterns = self.inner.get_recognized_patterns(None);
            for p in &patterns {
                if p.context.contains(&format!("'{pattern_type}'")) {
                    return Some(p.confidence);
                }
            }
        }
        None
    }

    fn __repr__(&self) -> String {
        let count = self.inner.get_recognized_patterns(None).len();
        format!("PatternDetector(recognized_patterns={count})")
    }
}

/// Convert a Python dict to HashMap<String, serde_json::Value>.
fn dict_to_hashmap(dict: &Bound<'_, PyDict>) -> PyResult<HashMap<String, serde_json::Value>> {
    let mut map = HashMap::new();
    for (k, v) in dict {
        let key: String = k.extract()?;
        map.insert(key, py_to_json(&v)?);
    }
    Ok(map)
}
