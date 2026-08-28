//! Configuration loader for the Nexus Teacher client.
//!
//! Loads settings from `.env` or system environment variables.

use serde::{Deserialize, Serialize};
use std::env;

/// Configuration for connecting to the 9router / OpenAI-compatible Teacher API.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TeacherConfig {
    /// Base URL for the API (e.g. `http://localhost:20128/v1`).
    pub api_url: String,
    /// Authorization Bearer token / API key.
    pub api_key: String,
    /// Target model identifier (e.g. `claude-opus-free` or `claude-3-opus-20240229`).
    pub model: String,
    /// Request timeout in seconds.
    pub timeout_secs: u64,
    /// If true, use local mock evaluation without network calls.
    pub mock_mode: bool,
}

impl Default for TeacherConfig {
    fn default() -> Self {
        Self {
            api_url: "http://localhost:20128/v1".to_string(),
            api_key: "default-key".to_string(),
            model: "claude-opus-free".to_string(),
            timeout_secs: 10,
            mock_mode: false,
        }
    }
}

impl TeacherConfig {
    /// Load teacher configuration from environment variables, attempting to read `.env` first.
    pub fn from_env() -> Self {
        let _ = dotenvy::dotenv();

        let api_url = env::var("NEXUS_TEACHER_API_URL")
            .unwrap_or_else(|_| "http://localhost:20128/v1".to_string());
        let api_key = env::var("NEXUS_TEACHER_API_KEY")
            .unwrap_or_else(|_| "default-key".to_string());
        let model = env::var("NEXUS_TEACHER_MODEL")
            .unwrap_or_else(|_| "claude-opus-free".to_string());
        let mock_mode = env::var("NEXUS_TEACHER_MOCK")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let timeout_secs = env::var("NEXUS_TEACHER_TIMEOUT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(10);

        Self {
            api_url,
            api_key,
            model,
            timeout_secs,
            mock_mode,
        }
    }

    /// Construct a mock configuration for deterministic testing.
    pub fn mock() -> Self {
        Self {
            mock_mode: true,
            ..Default::default()
        }
    }
}
