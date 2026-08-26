//! Asynchronous double-buffered prefetcher for MoE experts.
//!
//! Stages expert weight transfers from L2/L3 host memory into L1 VRAM in
//! parallel with GPU compute tasks.

use crate::warehouse::ExpertWarehouse;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use std::sync::Arc;
use tokio::sync::mpsc::{self, Receiver, Sender};

pub struct PrefetchRequest {
    pub expert_id: u64,
}

pub struct PrefetchResult<B: Backend> {
    pub expert_id: u64,
    pub weights: (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>),
}

/// Double-buffered async prefetcher.
pub struct AsyncPrefetcher<B: Backend> {
    warehouse: Arc<ExpertWarehouse<B>>,
    tx: Sender<PrefetchRequest>,
    rx: Receiver<PrefetchResult<B>>,
}

impl<B: Backend + 'static> AsyncPrefetcher<B> {
    pub fn new(warehouse: Arc<ExpertWarehouse<B>>, device: B::Device, buffer_size: usize) -> Self {
        let (req_tx, mut req_rx) = mpsc::channel::<PrefetchRequest>(buffer_size);
        let (res_tx, res_rx) = mpsc::channel::<PrefetchResult<B>>(buffer_size);

        let wh = Arc::clone(&warehouse);
        tokio::spawn(async move {
            while let Some(req) = req_rx.recv().await {
                if let Ok(weights) = wh.get_expert(req.expert_id, &device) {
                    let _ = res_tx
                        .send(PrefetchResult {
                            expert_id: req.expert_id,
                            weights,
                        })
                        .await;
                }
            }
        });

        Self {
            warehouse,
            tx: req_tx,
            rx: res_rx,
        }
    }

    /// Submit a prefetch request for an upcoming expert.
    pub async fn request(&self, expert_id: u64) -> anyhow::Result<()> {
        self.tx
            .send(PrefetchRequest { expert_id })
            .await
            .map_err(|e| anyhow::anyhow!("Prefetch request failed: {e}"))
    }

    /// Receive the next prefetched expert weights.
    pub async fn recv(&mut self) -> Option<PrefetchResult<B>> {
        self.rx.recv().await
    }

    /// Access the underlying warehouse.
    pub fn warehouse(&self) -> &Arc<ExpertWarehouse<B>> {
        &self.warehouse
    }
}
