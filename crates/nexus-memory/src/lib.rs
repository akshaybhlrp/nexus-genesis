//! Tiered expert storage: L1 (GPU VRAM), L2 (CPU RAM), L3 (SSD via memmap).
//!
//! Async double-buffered prefetcher for zero-stall expert swapping.

pub mod prefetcher;
pub mod warehouse;

pub use prefetcher::AsyncPrefetcher;
pub use warehouse::{ExpertWarehouse, SerializedExpert, SerializedTensor, WarehouseConfig};
