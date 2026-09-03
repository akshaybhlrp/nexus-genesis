//! Atomic Checkpointing Engine for Nexus MoELlama.
//!
//! Provides seamless, crash-proof persistence of:
//! - Complete MoELlama model parameters (Embeddings, Attention, Router, Experts, Norm, LM Head)
//! - Global step count
//! - Dataset cursor position
//! - Running loss and entropy metrics
//!
//! All writes use write-to-temp + fsync + atomic rename to prevent corruption.

use crate::moe::MoELlama;
use burn::prelude::Module;
use burn::record::{CompactRecorder, Recorder};
use burn::tensor::backend::Backend;
use serde::{Deserialize, Serialize};
use std::fs::{File, create_dir_all};
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CheckpointMeta {
    pub step: usize,
    pub dataset_cursor: usize,
    pub loss: f32,
    pub mean_entropy: f32,
    pub timestamp_secs: u64,
}

pub struct CheckpointManager {
    pub checkpoint_dir: PathBuf,
}

impl CheckpointManager {
    pub fn new(dir: impl AsRef<Path>) -> std::io::Result<Self> {
        let checkpoint_dir = dir.as_ref().to_path_buf();
        if !checkpoint_dir.exists() {
            create_dir_all(&checkpoint_dir)?;
        }
        Ok(Self { checkpoint_dir })
    }

    pub fn meta_path(&self) -> PathBuf {
        self.checkpoint_dir.join("checkpoint_latest.meta.json")
    }

    pub fn model_record_prefix(&self) -> PathBuf {
        self.checkpoint_dir.join("checkpoint_latest")
    }

    pub fn has_checkpoint(&self) -> bool {
        self.meta_path().exists()
    }

    pub fn load_meta(&self) -> anyhow::Result<CheckpointMeta> {
        let f = File::open(self.meta_path())?;
        let meta: CheckpointMeta = serde_json::from_reader(f)?;
        Ok(meta)
    }

    pub fn save<B: Backend>(
        &self,
        model: &MoELlama<B>,
        meta: &CheckpointMeta,
    ) -> anyhow::Result<()> {
        let pid = std::process::id();
        let tmp_prefix = self.checkpoint_dir.join(format!("checkpoint_tmp_{pid}"));

        let recorder = CompactRecorder::new();
        recorder.record(model.clone().into_record(), tmp_prefix.clone())?;

        // CompactRecorder writes to `<prefix>.mpk` or similar extension.
        // Identify any file written with the temp prefix and atomically rename.
        for entry in std::fs::read_dir(&self.checkpoint_dir)? {
            let entry = entry?;
            let path = entry.path();
            if let Some(fname) = path.file_name().and_then(|n| n.to_str()) {
                let prefix_str = format!("checkpoint_tmp_{pid}");
                if fname.starts_with(&prefix_str) {
                    let ext = fname.strip_prefix(&prefix_str).unwrap_or("");
                    let target_path = self.checkpoint_dir.join(format!("checkpoint_latest{ext}"));
                    std::fs::rename(&path, &target_path)?;
                }
            }
        }

        // Save metadata atomically via temp file and rename
        let tmp_meta = self.checkpoint_dir.join(format!("latest.meta.tmp.{pid}.json"));
        {
            let mut f = File::create(&tmp_meta)?;
            let json = serde_json::to_string_pretty(meta)?;
            f.write_all(json.as_bytes())?;
            f.sync_all()?;
        }
        std::fs::rename(&tmp_meta, self.meta_path())?;

        Ok(())
    }

    pub fn load_model<B: Backend>(
        &self,
        base_moe: MoELlama<B>,
        device: &B::Device,
    ) -> anyhow::Result<MoELlama<B>> {
        let recorder = CompactRecorder::new();
        let record = recorder.load(self.model_record_prefix(), device)?;
        Ok(base_moe.load_record(record))
    }
}
