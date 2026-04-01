//! Engine Core
//! 
//! The computational heart of the inference engine. This crate provides:
//! - Zero-copy memory arena for tensor operations
//! - Static computational graph with operator fusion
//! - Optimized mathematical kernels (SIMD-ready)
//! 
//! This crate is synchronous and has no async runtime dependencies.

#![allow(dead_code)]

pub mod arena;
pub mod graph;
pub mod ops;

use arena::{Arena, Section};
use graph::{Graph, Node, NodeMetadata, Op};
use ops::*;

/// The Inference Engine: synchronous graph executor
pub struct InferenceEngine {
    graph: Graph,
    arena: Arena<f32>,
}

impl InferenceEngine {
    /// Create a new inference engine with the given graph and memory configuration
    pub fn new(
        graph: Graph,
        weights_size: usize,
        activations_size: usize,
        io_size: usize,
    ) -> Self {
        Self {
            graph,
            arena: Arena::new(weights_size, activations_size, io_size),
        }
    }

    /// Load model weights from a byte buffer (typically from mmap)
    pub fn load_weights(&mut self, data: &[u8]) -> Result<(), arena::ArenaError> {
        self.arena.load_weights(data)
    }

    /// Execute the computational graph
    /// 
    /// This is the "hot path" - must be fast
    pub fn execute(&mut self) -> Result<(), ExecutionError> {
        // Clear activation arena before each run
        self.arena.clear_activations();

        // Execute each node in topological order
        for node in self.graph.nodes() {
            self.execute_node(node)?;
        }

        Ok(())
    }

    /// Execute a single node in the graph
    #[inline]
    fn execute_node(&mut self, node: &Node) -> Result<(), ExecutionError> {
        match node.op {
            Op::MatMul => self.compute_matmul(node),
            Op::ReLU => self.compute_relu(node),
            Op::GELU => self.compute_gelu(node),
            Op::Add => self.compute_add(node),
            Op::Softmax => self.compute_softmax(node),
            Op::LayerNorm => self.compute_layer_norm(node),
            Op::FusedLinearReLU => self.compute_fused_linear_relu(node),
            Op::FusedLinearGELU => self.compute_fused_linear_gelu(node),
            _ => Err(ExecutionError::UnsupportedOp(node.op)),
        }
    }

    fn compute_matmul(&mut self, node: &Node) -> Result<(), ExecutionError> {
        if let NodeMetadata::MatMul { m, k, n } = node.metadata {
            let input_idx = node.inputs[0];
            let weight_idx = node.inputs[1];
            let output_idx = node.output;

            // For now, use blocked matmul (will be replaced with SIMD version)
            let input = self.arena.view(Section::Activations, input_idx, m * k, vec![m, k]);
            let weights = self.arena.view(Section::Weights, weight_idx, k * n, vec![k, n]);
            let output = self.arena.view_mut(Section::Activations, output_idx, m * n, vec![m, n]);

            matmul_blocked(&input, &weights, output, m, k, n);
            Ok(())
        } else {
            Err(ExecutionError::InvalidMetadata)
        }
    }

    fn compute_relu(&mut self, node: &Node) -> Result<(), ExecutionError> {
        let input_idx = node.inputs[0];
        let output_idx = node.output;

        // Determine tensor size from input
        let input = self.arena.view(Section::Activations, input_idx, 1, vec![1]);
        let size = input.len();

        let input = self.arena.view(Section::Activations, input_idx, size, vec![size]);
        let output = self.arena.view_mut(Section::Activations, output_idx, size, vec![size]);

        relu(&input, output);
        Ok(())
    }

    fn compute_gelu(&mut self, node: &Node) -> Result<(), ExecutionError> {
        let input_idx = node.inputs[0];
        let output_idx = node.output;

        let input = self.arena.view(Section::Activations, input_idx, 1, vec![1]);
        let size = input.len();

        let input = self.arena.view(Section::Activations, input_idx, size, vec![size]);
        let output = self.arena.view_mut(Section::Activations, output_idx, size, vec![size]);

        gelu(&input, output);
        Ok(())
    }

    fn compute_add(&mut self, node: &Node) -> Result<(), ExecutionError> {
        let a_idx = node.inputs[0];
        let b_idx = node.inputs[1];
        let output_idx = node.output;

        let a = self.arena.view(Section::Activations, a_idx, 1, vec![1]);
        let size = a.len();

        let a = self.arena.view(Section::Activations, a_idx, size, vec![size]);
        let b = self.arena.view(Section::Activations, b_idx, size, vec![size]);
        let output = self.arena.view_mut(Section::Activations, output_idx, size, vec![size]);

        add(&a, &b, output);
        Ok(())
    }

    fn compute_softmax(&mut self, node: &Node) -> Result<(), ExecutionError> {
        let input_idx = node.inputs[0];
        let output_idx = node.output;

        let input = self.arena.view(Section::Activations, input_idx, 1, vec![1]);
        let size = input.len();

        let input = self.arena.view(Section::Activations, input_idx, size, vec![size]);
        let output = self.arena.view_mut(Section::Activations, output_idx, size, vec![size]);

        softmax(&input, output);
        Ok(())
    }

    fn compute_layer_norm(&mut self, node: &Node) -> Result<(), ExecutionError> {
        if let NodeMetadata::LayerNorm { epsilon } = node.metadata {
            let input_idx = node.inputs[0];
            let output_idx = node.output;

            let input = self.arena.view(Section::Activations, input_idx, 1, vec![1]);
            let size = input.len();

            let input = self.arena.view(Section::Activations, input_idx, size, vec![size]);
            let output = self.arena.view_mut(Section::Activations, output_idx, size, vec![size]);

            layer_norm(&input, output, epsilon);
            Ok(())
        } else {
            Err(ExecutionError::InvalidMetadata)
        }
    }

    fn compute_fused_linear_relu(&mut self, node: &Node) -> Result<(), ExecutionError> {
        if let NodeMetadata::MatMul { m, k, n } = node.metadata {
            let input_idx = node.inputs[0];
            let weight_idx = node.inputs[1];
            let output_idx = node.output;

            let input = self.arena.view(Section::Activations, input_idx, m * k, vec![m, k]);
            let weights = self.arena.view(Section::Weights, weight_idx, k * n, vec![k, n]);
            let output = self.arena.view_mut(Section::Activations, output_idx, m * n, vec![m, n]);

            fused_linear_relu(&input, &weights, output, m, k, n);
            Ok(())
        } else {
            Err(ExecutionError::InvalidMetadata)
        }
    }

    fn compute_fused_linear_gelu(&mut self, _node: &Node) -> Result<(), ExecutionError> {
        // TODO: Implement fused linear + GELU
        Err(ExecutionError::NotImplemented)
    }

    /// Get a reference to the arena (for inspection/debugging)
    pub fn arena(&self) -> &Arena<f32> {
        &self.arena
    }

    /// Get a mutable reference to the arena (for I/O operations)
    pub fn arena_mut(&mut self) -> &mut Arena<f32> {
        &mut self.arena
    }

    /// Get graph statistics
    pub fn graph_stats(&self) -> graph::GraphStats {
        self.graph.stats()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ExecutionError {
    #[error("Unsupported operation: {0:?}")]
    UnsupportedOp(Op),
    
    #[error("Invalid node metadata")]
    InvalidMetadata,
    
    #[error("Not implemented")]
    NotImplemented,
}

#[cfg(test)]
mod tests {
    use super::*;
    use graph::GraphBuilder;

    #[test]
    fn test_simple_inference() {
        // Build a simple graph: Input -> MatMul -> ReLU -> Output
        let graph = GraphBuilder::new(10)
            .add_matmul(0, 1, 2, 2, 3, 2)  // [2x3] * [3x2] = [2x2]
            .add_relu(2, 3)
            .build();

        let mut engine = InferenceEngine::new(graph, 100, 200, 50);

        // Execute
        let result = engine.execute();
        assert!(result.is_ok());
    }

    #[test]
    fn test_fused_execution() {
        // Build graph that should be fused
        let graph = GraphBuilder::new(10)
            .add_matmul(0, 1, 2, 2, 3, 2)
            .add_relu(2, 3)
            .optimize();  // This should fuse MatMul + ReLU

        let stats = graph.stats();
        assert_eq!(stats.num_nodes, 1, "Graph should have 1 fused node");

        let mut engine = InferenceEngine::new(graph, 100, 200, 50);
        let result = engine.execute();
        assert!(result.is_ok());
    }
}
