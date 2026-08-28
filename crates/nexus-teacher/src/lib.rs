//! Nexus Teacher: external validation via 9router API (OpenAI-compatible).
//!
//! Queries a remote LLM (e.g. Claude Opus via 9router) on high-entropy
//! outlier batches to guide mutation rate and resistance updates.
//! API config loaded from environment variables (`.env`).

pub mod config;
pub mod validator;

pub use config::TeacherConfig;
pub use validator::{TeacherFeedback, TeacherValidator};
