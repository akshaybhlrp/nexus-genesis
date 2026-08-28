//! Alignment / Reward-model verification for the Teacher validator:
//! zero-shot prompt grading, few-shot caching, score bounds, negative-angle
//! cases, and red-teaming-style malicious-input rejection semantics.

use nexus_teacher::validator::TeacherValidator;

/// Mock teacher must return a score in [0,1] and cache results.
#[tokio::test]
async fn mock_validator_score_bounded_and_cached() {
    let v = TeacherValidator::new_mock(0.9);
    let fb = v.validate("prompt", "output").await.unwrap();
    assert!((0.0..=1.0).contains(&fb.score), "score out of bounds: {}", fb.score);
    assert!(!fb.cached, "first call not cached");
    assert_eq!(v.cache_len(), 1);

    // Second identical call hits cache.
    let fb2 = v.validate("prompt", "output").await.unwrap();
    assert!(fb2.cached, "second call must be served from cache");
    assert_eq!(fb2.score, fb.score);
}

/// Mock validator clamps out-of-range scores to [0,1].
#[tokio::test]
async fn mock_validator_clamps_extreme_scores() {
    let v = TeacherValidator::new_mock(5.0);
    let fb = v.validate("a", "b").await.unwrap();
    assert!((fb.score - 1.0).abs() < 1e-5, "5.0 clamped to 1.0, got {}", fb.score);

    let v2 = TeacherValidator::new_mock(-3.0);
    let fb2 = v2.validate("a", "b").await.unwrap();
    assert!((fb2.score).abs() < 1e-5, "-3.0 clamped to 0.0, got {}", fb2.score);
}

/// Cache must be keyed by (prompt, output) — same prompt, different output is
/// a distinct cache entry.
#[tokio::test]
async fn cache_is_keyed_on_prompt_and_output() {
    let v = TeacherValidator::new_mock(0.6);
    let _ = v.validate("p", "good ans").await.unwrap();
    let _ = v.validate("p", "bad answer!!").await.unwrap();
    assert_eq!(v.cache_len(), 2, "different outputs must not share a cache line");
}

/// Negative case: an unknown/invalid output still yields a bounded mock score
/// (no panic, no NaN).
#[tokio::test]
async fn mock_validator_tolerates_garbage_input() {
    let v = TeacherValidator::new_mock(0.5);
    let fb = v.validate("", "\u{1}\u{2}binary-junk").await.unwrap();
    assert!(fb.score.is_finite());
    assert!((0.0..=1.0).contains(&fb.score));
}

/// Red-teaming mode: a mock validator can be configured to score harmful
/// content low, demonstrating the alignment signal is actionable (framework
/// test: the plumbing produces a low score rather than failing).
#[tokio::test]
async fn red_teaming_low_score_signal() {
    // A stricter "teacher" (low fixed score) models a safety-orientated
    // validator. The harness must propagate that score so training can react.
    let strict = TeacherValidator::new_mock(0.1);
    let fb = strict
        .validate("harmful instruction", "malicious completion")
        .await
        .unwrap();
    assert!(fb.score <= 0.5, "red-team-triggering content must score low, got {}", fb.score);
}

/// Cache is cleared on demand.
#[tokio::test]
async fn clear_cache_empties_entries() {
    let v = TeacherValidator::new_mock(0.7);
    for i in 0..50 {
        let _ = v.validate(&format!("p{i}"), "out").await.unwrap();
    }
    assert_eq!(v.cache_len(), 50);
    v.clear_cache();
    assert_eq!(v.cache_len(), 0);
}

/// Concurrent access to a mock validator is safe (RwLock) — moderate load.
#[tokio::test]
async fn concurrent_mock_validation_is_safe() {
    let v = std::sync::Arc::new(TeacherValidator::new_mock(0.8));
    let mut handles = Vec::new();
    for i in 0..32 {
        let v = std::sync::Arc::clone(&v);
        handles.push(tokio::spawn(async move {
            let fb = v.validate(&format!("p{i}"), "out").await.unwrap();
            assert!((fb.score - 0.8).abs() < 1e-5);
        }));
    }
    for h in handles {
        h.await.unwrap();
    }
    assert!((v.cache_len()) >= 1);
}
