//! Teacher validator client for remote LLM scoring.
//!
//! Queries an OpenAI-compatible API (9router -> Claude Opus) to grade
//! generated model outputs on high-entropy outlier batches.

use crate::config::TeacherConfig;
use reqwest::Client;
use serde_json::json;
use std::collections::HashMap;
use std::sync::RwLock;
use std::time::Duration;

/// Feedback returned from the Teacher.
#[derive(Clone, Debug)]
pub struct TeacherFeedback {
    /// Quality/accuracy score in range [0.0, 1.0].
    pub score: f32,
    /// Raw explanation/reasoning from the teacher if available.
    pub reasoning: String,
    /// Whether this score came from the semantic cache.
    pub cached: bool,
}

/// External teacher validator with prompt-response semantic caching.
pub struct TeacherValidator {
    config: TeacherConfig,
    client: Client,
    cache: RwLock<HashMap<String, f32>>,
    mock_score: Option<f32>,
}

impl TeacherValidator {
    /// Create a new `TeacherValidator` with given configuration.
    pub fn new(config: TeacherConfig) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .unwrap_or_default();

        Self {
            config,
            client,
            cache: RwLock::new(HashMap::new()),
            mock_score: None,
        }
    }

    /// Create a mock validator with a fixed response score for testing.
    pub fn new_mock(fixed_score: f32) -> Self {
        Self {
            config: TeacherConfig::mock(),
            client: Client::default(),
            cache: RwLock::new(HashMap::new()),
            mock_score: Some(fixed_score.clamp(0.0, 1.0)),
        }
    }

    /// Check cache or query the teacher API for a rating of `(prompt, output)`.
    pub async fn validate(&self, prompt: &str, output: &str) -> anyhow::Result<TeacherFeedback> {
        let cache_key = format!("{prompt}\t{output}");

        // 1. Check cache
        {
            let cache = self.cache.read().unwrap();
            if let Some(&score) = cache.get(&cache_key) {
                return Ok(TeacherFeedback {
                    score,
                    reasoning: "cached".to_string(),
                    cached: true,
                });
            }
        }

        // 2. Handle mock mode
        if self.config.mock_mode || self.mock_score.is_some() {
            let score = self.mock_score.unwrap_or(0.8);
            let mut cache = self.cache.write().unwrap();
            cache.insert(cache_key, score);
            return Ok(TeacherFeedback {
                score,
                reasoning: "mock evaluation".to_string(),
                cached: false,
            });
        }

        // 3. Make HTTP request to OpenAI-compatible endpoint
        let endpoint = format!("{}/chat/completions", self.config.api_url.trim_end_matches('/'));
        let system_prompt = "You are an automated evaluator. Grade the model response for coherence, factual accuracy, and semantic quality. Output ONLY a single floating-point number between 0.0 and 1.0 where 1.0 is perfect quality and 0.0 is complete failure.";

        let user_content = format!("Input Context:\n{prompt}\n\nModel Response:\n{output}");

        let request_body = json!({
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "max_tokens": 16,
            "temperature": 0.0
        });

        let resp = self
            .client
            .post(&endpoint)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await?;

        let status = resp.status();
        if !status.is_success() {
            let err_text = resp.text().await.unwrap_or_default();
            anyhow::bail!("Teacher API returned status {status}: {err_text}");
        }

        let json_resp: serde_json::Value = resp.json().await?;
        let raw_content = json_resp["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("0.5")
            .trim();

        let score = parse_score(raw_content);

        // 4. Update cache
        {
            let mut cache = self.cache.write().unwrap();
            cache.insert(cache_key, score);
        }

        Ok(TeacherFeedback {
            score,
            reasoning: raw_content.to_string(),
            cached: false,
        })
    }

    /// Clear cache.
    pub fn clear_cache(&self) {
        let mut cache = self.cache.write().unwrap();
        cache.clear();
    }

    /// Number of cached prompt-response pairs.
    pub fn cache_len(&self) -> usize {
        self.cache.read().unwrap().len()
    }
}

/// Extract float score between 0.0 and 1.0 from LLM string.
/// Extract the first numeric score from a teacher response string and clamp
/// it to [0, 1]. Returns the fallback `0.5` when no number is present.
///
/// Robust to arbitrary surrounding text: handles
/// `"0.75"`, `"Score: 0.92\n..."`, `"Rating: 0.45/1.0"`, `"0.80 overall"`,
/// and out-of-range / malformed input.
fn parse_score(text: &str) -> f32 {
    if let Ok(val) = text.trim().parse::<f32>() {
        return val.clamp(0.0, 1.0);
    }
    // Scan for the first embedded numeric run (e.g. `0.45` inside
    // `Rating: 0.45/1.0`), honoring a leading sign/dot.
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
        // Consume the run of digits / separators.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_score() {
        assert_eq!(parse_score("0.95"), 0.95);
        assert_eq!(parse_score("1.0"), 1.0);
        assert_eq!(parse_score("0.0"), 0.0);
        assert_eq!(parse_score("Score: 0.82\nReasoning: good"), 0.82);
        assert_eq!(parse_score("invalid garbage"), 0.5);
    }

    #[tokio::test]
    async fn test_mock_validator_caching() -> anyhow::Result<()> {
        let validator = TeacherValidator::new_mock(0.75);
        assert_eq!(validator.cache_len(), 0);

        let fb1 = validator.validate("What is 2+2?", "4").await?;
        assert_eq!(fb1.score, 0.75);
        assert!(!fb1.cached);
        assert_eq!(validator.cache_len(), 1);

        let fb2 = validator.validate("What is 2+2?", "4").await?;
        assert_eq!(fb2.score, 0.75);
        assert!(fb2.cached);

        let fb3 = validator.validate("What is 3+3?", "6").await?;
        assert_eq!(fb3.score, 0.75);
        assert!(!fb3.cached);
        assert_eq!(validator.cache_len(), 2);

        Ok(())
    }
}
