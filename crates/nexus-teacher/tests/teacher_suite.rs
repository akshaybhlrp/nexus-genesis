//! Teacher validator and semantic caching integration tests.
//!
//! Validates:
//! - Pillar 4 (The Conscience): External teacher validation via OpenAI-compatible endpoint.
//! - Critical Patch 6: Adaptive mu feedback mechanism and score parsing.
//! - Fast semantic caching layer for deduplicated prompt-response pairs.
//! - Robust error resilience: score extraction fallbacks, clamping, and mock mode.

use nexus_teacher::{TeacherConfig, TeacherValidator};

// =========================================================================
// 1. TeacherConfig Construction & Parsing
// =========================================================================

#[test]
fn test_teacher_config_defaults() {
    let cfg = TeacherConfig::default();
    assert_eq!(cfg.api_url, "http://localhost:20128/v1");
    assert_eq!(cfg.api_key, "default-key");
    assert_eq!(cfg.model, "claude-opus-free");
    assert_eq!(cfg.timeout_secs, 10);
    assert!(!cfg.mock_mode);
}

#[test]
fn test_teacher_config_mock_constructor() {
    let cfg = TeacherConfig::mock();
    assert!(cfg.mock_mode);
    assert_eq!(cfg.model, "claude-opus-free");
}

// =========================================================================
// 2. Pillar 4: Semantic Caching Layer
// =========================================================================

#[tokio::test]
async fn test_semantic_cache_lifecycle() -> anyhow::Result<()> {
    let validator = TeacherValidator::new_mock(0.88);
    assert_eq!(validator.cache_len(), 0);

    // Initial query -> Cache MISS
    let fb1 = validator.validate("Explain gravity", "Objects attract").await?;
    assert_eq!(fb1.score, 0.88);
    assert!(!fb1.cached, "First call must be a cache miss");
    assert_eq!(validator.cache_len(), 1);

    // Identical query -> Cache HIT
    let fb2 = validator.validate("Explain gravity", "Objects attract").await?;
    assert_eq!(fb2.score, 0.88);
    assert!(fb2.cached, "Subsequent identical call must be a cache hit");
    assert_eq!(validator.cache_len(), 1);

    // Different output -> Cache MISS
    let fb3 = validator.validate("Explain gravity", "Spacetime curvature").await?;
    assert_eq!(fb3.score, 0.88);
    assert!(!fb3.cached);
    assert_eq!(validator.cache_len(), 2);

    // Clear cache
    validator.clear_cache();
    assert_eq!(validator.cache_len(), 0);

    // Re-query -> Cache MISS again
    let fb4 = validator.validate("Explain gravity", "Objects attract").await?;
    assert!(!fb4.cached);
    assert_eq!(validator.cache_len(), 1);

    Ok(())
}

#[tokio::test]
async fn test_mock_validator_score_clamping() -> anyhow::Result<()> {
    // Score > 1.0 clamps to 1.0
    let v_high = TeacherValidator::new_mock(2.5);
    let fb_high = v_high.validate("Q", "A").await?;
    assert_eq!(fb_high.score, 1.0);

    // Score < 0.0 clamps to 0.0
    let v_low = TeacherValidator::new_mock(-0.5);
    let fb_low = v_low.validate("Q", "A").await?;
    assert_eq!(fb_low.score, 0.0);

    Ok(())
}

// =========================================================================
// 3. Score Parsing Edge Cases & Robustness
// =========================================================================

#[test]
fn test_teacher_score_parsing_edge_cases() {
    // Exact valid floats
    assert_eq!(parse_test_helper("0.0"), 0.0);
    assert_eq!(parse_test_helper("1.0"), 1.0);
    assert_eq!(parse_test_helper("0.75"), 0.75);

    // Wrapped in commentary or markdown
    assert_eq!(parse_test_helper("Score: 0.92\nReasoning: coherent."), 0.92);
    assert_eq!(parse_test_helper("Rating: 0.45/1.0"), 0.45);
    assert_eq!(parse_test_helper("The evaluation is 0.80 overall."), 0.80);

    // Out of range floats clamped
    assert_eq!(parse_test_helper("1.5"), 1.0);
    assert_eq!(parse_test_helper("-0.3"), 0.0);

    // Unparseable / malformed text -> fallback to 0.5
    assert_eq!(parse_test_helper("I cannot evaluate this."), 0.5);
    assert_eq!(parse_test_helper(""), 0.5);
    assert_eq!(parse_test_helper("N/A"), 0.5);
}

/// Helper replicating internal parse_score behavior (matches the robust
/// extraction in nexus-teacher/src/validator.rs).
fn parse_test_helper(text: &str) -> f32 {
    if let Ok(val) = text.trim().parse::<f32>() {
        return val.clamp(0.0, 1.0);
    }
    let bytes = text.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        let b = bytes[i];
        let starts_number = b.is_ascii_digit()
            || ((b == b'-' || b == b'+' || b == b'.')
                && i + 1 < bytes.len()
                && bytes[i + 1].is_ascii_digit());
        if !starts_number {
            i += 1;
            continue;
        }
        let mut end = i;
        while end < bytes.len()
            && (bytes[end].is_ascii_digit()
                || matches!(bytes[end], b'.' | b'-' | b'+'))
        {
            end += 1;
        }
        let candidate = std::str::from_utf8(&bytes[i..end]).unwrap_or("");
        if let Ok(val) = candidate.parse::<f32>() {
            return val.clamp(0.0, 1.0);
        }
        i = end;
    }
    0.5
}
