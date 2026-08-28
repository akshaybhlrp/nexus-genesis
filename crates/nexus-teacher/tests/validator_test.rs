use nexus_teacher::{TeacherConfig, TeacherValidator};

#[tokio::test]
async fn test_teacher_validation_flow_and_cache() -> anyhow::Result<()> {
    let cfg = TeacherConfig {
        api_url: "http://localhost:20128/v1".to_string(),
        api_key: "test-key".to_string(),
        model: "claude-opus-free".to_string(),
        timeout_secs: 5,
        mock_mode: true,
    };

    let validator = TeacherValidator::new(cfg);
    let fb1 = validator.validate("Input prompt", "Generated output").await?;
    assert!(fb1.score >= 0.0 && fb1.score <= 1.0);
    assert!(!fb1.cached);

    // Second call should be cached
    let fb2 = validator.validate("Input prompt", "Generated output").await?;
    assert_eq!(fb1.score, fb2.score);
    assert!(fb2.cached);

    // Test fixed score mock
    let fixed_validator = TeacherValidator::new_mock(0.92);
    let fb_fixed = fixed_validator.validate("Prompt", "Output").await?;
    assert_eq!(fb_fixed.score, 0.92);

    Ok(())
}
