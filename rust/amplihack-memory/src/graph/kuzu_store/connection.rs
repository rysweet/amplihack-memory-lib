use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::KuzuGraphStore;

impl KuzuGraphStore {
    /// Execute a Cypher query with a PyDict of parameters.
    pub(crate) fn execute_cypher(
        &self,
        py: Python<'_>,
        cypher: &str,
        params: &Bound<'_, PyDict>,
    ) -> crate::Result<PyObject> {
        let conn = self.py_conn.bind(py);
        conn.call_method("execute", (cypher, params), None)
            .map(|r| r.unbind())
            .map_err(|e| {
                crate::MemoryError::Storage(format!(
                    "Cypher execution failed: {e}\nQuery: {cypher}"
                ))
            })
    }

    /// Execute a Cypher query without parameters.
    pub(crate) fn execute_no_params(
        &self,
        py: Python<'_>,
        cypher: &str,
    ) -> crate::Result<PyObject> {
        let conn = self.py_conn.bind(py);
        conn.call_method1("execute", (cypher,))
            .map(|r| r.unbind())
            .map_err(|e| {
                crate::MemoryError::Storage(format!(
                    "Cypher execution failed: {e}\nQuery: {cypher}"
                ))
            })
    }
}
