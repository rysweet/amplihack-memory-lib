#[cfg(test)]
mod tests {
    use crate::backends::base::MemoryBackend;
    use crate::backends::sqlite_backend::{escape_fts5_query, SqliteBackend};
    use crate::experience::{Experience, ExperienceType};

    use chrono::Utc;
    use tempfile::NamedTempFile;

    fn test_backend() -> (SqliteBackend, NamedTempFile) {
        let tmp = NamedTempFile::new().unwrap();
        let backend = SqliteBackend::new(tmp.path(), "test-agent", 100, true).unwrap();
        (backend, tmp)
    }

    fn test_experience() -> Experience {
        Experience::new(
            ExperienceType::Success,
            "test context".into(),
            "test outcome".into(),
            0.9,
        )
        .unwrap()
    }

    #[test]
    fn test_store_and_retrieve() {
        let (mut backend, _tmp) = test_backend();
        let exp = test_experience();
        let id = backend.store_experience(&exp).unwrap();
        assert!(!id.is_empty());

        let results = backend.retrieve_experiences(Some(10), None, 0.0).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].context, "test context");
    }

    #[test]
    fn test_search_fts() {
        let (mut backend, _tmp) = test_backend();
        let exp = test_experience();
        backend.store_experience(&exp).unwrap();

        let results = MemoryBackend::search(&backend, "test", None, 0.0, 10).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_statistics() {
        let (mut backend, _tmp) = test_backend();
        let exp = test_experience();
        backend.store_experience(&exp).unwrap();

        let stats = MemoryBackend::get_statistics(&backend).unwrap();
        assert_eq!(stats.total_experiences, 1);
    }

    #[test]
    fn test_filter_by_type() {
        let (mut backend, _tmp) = test_backend();
        let success =
            Experience::new(ExperienceType::Success, "ctx".into(), "out".into(), 0.9).unwrap();
        let failure =
            Experience::new(ExperienceType::Failure, "ctx2".into(), "out2".into(), 0.8).unwrap();
        backend.store_experience(&success).unwrap();
        backend.store_experience(&failure).unwrap();

        let results = backend
            .retrieve_experiences(Some(10), Some(ExperienceType::Success), 0.0)
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].experience_type, ExperienceType::Success);
    }

    #[test]
    fn test_min_confidence_filter() {
        let (mut backend, _tmp) = test_backend();
        let high = Experience::new(
            ExperienceType::Success,
            "high".into(),
            "high out".into(),
            0.9,
        )
        .unwrap();
        let low = Experience::new(
            ExperienceType::Success,
            "low conf".into(),
            "low out".into(),
            0.3,
        )
        .unwrap();
        backend.store_experience(&high).unwrap();
        backend.store_experience(&low).unwrap();

        let results = backend.retrieve_experiences(Some(10), None, 0.5).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].context, "high");
    }

    #[test]
    fn test_cleanup_auto_compress() {
        let (mut backend, _tmp) = test_backend();
        // Store an experience with a timestamp older than 30 days
        let old_ts = Utc::now() - chrono::Duration::days(60);
        let exp = Experience::with_timestamp(
            ExperienceType::Success,
            "old context".into(),
            "old outcome".into(),
            0.9,
            old_ts,
        )
        .unwrap();
        backend.store_experience(&exp).unwrap();

        // Also store a recent experience
        let recent = test_experience();
        backend.store_experience(&recent).unwrap();

        backend.cleanup(true, None, None).unwrap();

        let stats = MemoryBackend::get_statistics(&backend).unwrap();
        assert_eq!(stats.total_experiences, 2);
        // Only the old one should be compressed
        assert_eq!(stats.compressed_experiences, 1);
    }

    #[test]
    fn test_cleanup_max_age_days() {
        let (mut backend, _tmp) = test_backend();
        let old_ts = Utc::now() - chrono::Duration::days(100);
        let old_exp = Experience::with_timestamp(
            ExperienceType::Success,
            "ancient context".into(),
            "ancient outcome".into(),
            0.9,
            old_ts,
        )
        .unwrap();
        backend.store_experience(&old_exp).unwrap();

        let recent = test_experience();
        backend.store_experience(&recent).unwrap();

        // Delete experiences older than 50 days
        backend.cleanup(false, Some(50), None).unwrap();

        let results = backend.retrieve_experiences(Some(10), None, 0.0).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].context, "test context");
    }

    #[test]
    fn test_cleanup_max_experiences() {
        let (mut backend, _tmp) = test_backend();
        // Store 5 experiences with staggered timestamps
        for i in 0..5 {
            let ts = Utc::now() - chrono::Duration::seconds(50 - i);
            let exp = Experience::with_timestamp(
                ExperienceType::Success,
                format!("ctx {i}"),
                format!("out {i}"),
                0.5,
                ts,
            )
            .unwrap();
            backend.store_experience(&exp).unwrap();
        }

        let before = backend.retrieve_experiences(Some(100), None, 0.0).unwrap();
        assert_eq!(before.len(), 5);

        // Trim to max 3 experiences
        backend.cleanup(false, None, Some(3)).unwrap();

        let after = backend.retrieve_experiences(Some(100), None, 0.0).unwrap();
        assert!(after.len() <= 3, "expected at most 3, got {}", after.len());
    }

    // ====================================================================
    // #31: FTS5 query escaping
    // ====================================================================

    #[test]
    fn test_escape_fts5_query_basic() {
        let escaped = escape_fts5_query("hello world");
        assert_eq!(escaped, r#""hello" "world""#);
    }

    #[test]
    fn test_escape_fts5_query_operators() {
        let escaped = escape_fts5_query("AND OR NOT NEAR");
        assert_eq!(escaped, r#""AND" "OR" "NOT" "NEAR""#);
    }

    #[test]
    fn test_escape_fts5_query_wildcard() {
        let escaped = escape_fts5_query("test*");
        assert_eq!(escaped, r#""test*""#);
    }

    #[test]
    fn test_escape_fts5_query_with_quotes() {
        let escaped = escape_fts5_query(r#"say "hello""#);
        assert_eq!(escaped, r#""say" """hello""""#);
    }

    #[test]
    fn test_escape_fts5_query_empty() {
        let escaped = escape_fts5_query("");
        assert_eq!(escaped, "");
    }

    #[test]
    fn test_search_with_fts5_operators() {
        let tmp = tempfile::TempDir::new().unwrap();
        let db_path = tmp.path().join("fts5_test.db");
        let mut backend = SqliteBackend::new(&db_path, "fts5-agent", 100, true).unwrap();

        let exp = Experience::new(
            ExperienceType::Success,
            "using AND operator in context".into(),
            "outcome with NOT keyword".into(),
            0.9,
        )
        .unwrap();
        backend.store_experience(&exp).unwrap();

        // Searching for "AND" should not cause FTS5 syntax error
        let results = MemoryBackend::search(&backend, "AND", None, 0.0, 10).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_qa_retrieve_with_logging() {
        let (mut backend, _tmp) = test_backend();
        let exp = test_experience();
        backend.store_experience(&exp).unwrap();
        let results = backend.retrieve_experiences(Some(10), None, 0.0).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_qa_statistics_propagation() {
        let (mut backend, _tmp) = test_backend();
        let exp = test_experience();
        backend.store_experience(&exp).unwrap();
        let stats = backend.get_statistics().unwrap();
        assert_eq!(stats.total_experiences, 1);
        assert_eq!(stats.compressed_experiences, 0);
    }

    #[test]
    fn test_qa_cleanup_propagation() {
        let (mut backend, _tmp) = test_backend();
        for _ in 0..5 {
            backend.store_experience(&test_experience()).unwrap();
        }
        let result = backend.cleanup(false, None, Some(3));
        assert!(result.is_ok());
    }
}
