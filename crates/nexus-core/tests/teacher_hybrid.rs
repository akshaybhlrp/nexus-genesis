use burn::backend::Autodiff;
use nexus_core::hybrid::{train_hybrid, HybridConfig};
use nexus_core::model::LlamaConfig;
use nexus_core::moe::{upcycle_dense, RouterConfig};
use nexus_core::stream::synthetic_stream;
use nexus_teacher::TeacherValidator;
use std::sync::Arc;

type TestBackend = Autodiff<burn::backend::Wgpu>;

#[test]
fn test_hybrid_training_with_teacher_feedback() {
    let device = Default::default();
    let model_cfg = LlamaConfig::new(256, 32, 4, 1)
        .with_max_seq_len(128)
        .with_d_ff(64);
    let dense = model_cfg.init::<TestBackend>(&device);
    let router_cfg = RouterConfig::new(4);
    let moe = upcycle_dense(&dense, &router_cfg);

    // Mock teacher with low score (0.2) to trigger exploration & LR dampening
    let teacher = Arc::new(TeacherValidator::new_mock(0.2));
    let mut hybrid_cfg = HybridConfig::default();
    hybrid_cfg.entropy_threshold = 0.01; // Force trigger teacher check
    hybrid_cfg.teacher = Some(teacher);

    let stream = synthetic_stream(20 * 4, 256);
    let (_trained, metrics) = train_hybrid(moe, stream, 5, 4, hybrid_cfg);

    assert_eq!(metrics.len(), 5);
    let queried = metrics.iter().any(|m| m.teacher_queried);
    assert!(queried, "Teacher should have been queried on high entropy");

    let teacher_scores: Vec<_> = metrics.iter().filter_map(|m| m.teacher_score).collect();
    assert!(!teacher_scores.is_empty());
    assert_eq!(teacher_scores[0], 0.2);
}
