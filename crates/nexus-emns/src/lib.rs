//! EMNS (Evolutionary Mutation with Natural Selection) kernel.
//!
//! CubeCL compute kernel for in-place weight mutation across
//! CUDA, Metal, and Vulkan/WGSL backends.
//!
//! Phase 3 delivers the CPU-side mutator (`mutator` module). The CubeCL
//! kernel is a stretch goal for when GPU-side mutation is needed.

pub mod mutator;
