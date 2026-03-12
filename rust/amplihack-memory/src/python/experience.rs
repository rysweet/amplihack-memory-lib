//! PyO3 wrapper for the `Experience` struct.
#![allow(clippy::useless_conversion)]

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::experience::Experience;

use super::helpers::{hashmap_to_pydict, mem_err, parse_experience_type, pydict_to_hashmap};

/// A single experience record wrapping the Rust `Experience` struct.
#[pyclass(name = "Experience")]
#[derive(Clone)]
pub struct PyExperience {
    pub(crate) inner: Experience,
}

#[pymethods]
impl PyExperience {
    /// Create a new Experience.
    ///
    /// Args:
    ///     experience_type: One of "success", "failure", "pattern", "insight"
    ///     context: Context string
    ///     outcome: Outcome string
    ///     confidence: Float 0.0–1.0
    ///     tags: Optional list of tag strings
    ///     metadata: Optional dict of metadata
    #[new]
    #[pyo3(signature = (experience_type, context, outcome, confidence, tags=None, metadata=None))]
    fn new(
        experience_type: &str,
        context: String,
        outcome: String,
        confidence: f64,
        tags: Option<Vec<String>>,
        metadata: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let et = parse_experience_type(experience_type)?;
        let mut exp = Experience::new(et, context, outcome, confidence).map_err(mem_err)?;
        if let Some(t) = tags {
            exp.tags = t;
        }
        if let Some(m) = metadata {
            exp.metadata = pydict_to_hashmap(Some(m))?;
        }
        Ok(Self { inner: exp })
    }

    #[getter]
    fn id(&self) -> &str {
        &self.inner.experience_id
    }

    #[getter]
    fn experience_type(&self) -> &str {
        self.inner.experience_type.as_str()
    }

    #[getter]
    fn context(&self) -> &str {
        &self.inner.context
    }

    #[getter]
    fn outcome(&self) -> &str {
        &self.inner.outcome
    }

    #[getter]
    fn confidence(&self) -> f64 {
        self.inner.confidence
    }

    #[getter]
    fn tags(&self) -> Vec<String> {
        self.inner.tags.clone()
    }

    #[getter]
    fn timestamp(&self) -> String {
        self.inner.timestamp.to_rfc3339()
    }

    #[getter]
    fn metadata<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        hashmap_to_pydict(py, &self.inner.metadata)
    }

    /// Serialize to a Python dict.
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let d = PyDict::new_bound(py);
        d.set_item("id", &self.inner.experience_id)?;
        d.set_item("experience_type", self.inner.experience_type.as_str())?;
        d.set_item("context", &self.inner.context)?;
        d.set_item("outcome", &self.inner.outcome)?;
        d.set_item("confidence", self.inner.confidence)?;
        d.set_item("tags", self.inner.tags.to_object(py))?;
        d.set_item("timestamp", self.inner.timestamp.to_rfc3339())?;
        d.set_item("metadata", hashmap_to_pydict(py, &self.inner.metadata)?)?;
        Ok(d.to_object(py))
    }

    fn __repr__(&self) -> String {
        format!(
            "Experience(id={}, type={}, confidence={})",
            self.inner.experience_id,
            self.inner.experience_type.as_str(),
            self.inner.confidence
        )
    }
}

impl PyExperience {
    pub(crate) fn from_rust(exp: Experience) -> Self {
        Self { inner: exp }
    }
}
