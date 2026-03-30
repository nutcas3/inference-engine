//! Computational Graph
//! 
//! Static DAG (Directed Acyclic Graph) representation of the neural network.
//! Once loaded from GGUF/ONNX, the graph is immutable during inference.

use std::fmt;

/// Operation types supported by the inference engine
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    /// Matrix multiplication: Y = X * W
    MatMul,
    /// Element-wise addition: Y = X + b
    Add,
    /// ReLU activation: Y = max(0, X)
    ReLU,
    /// GELU activation (Gaussian Error Linear Unit)
    GELU,
    /// Softmax normalization
    Softmax,
    /// Layer normalization
    LayerNorm,
    /// Fused operation: Linear + ReLU
    FusedLinearReLU,
    /// Fused operation: Linear + GELU
    FusedLinearGELU,
    /// Attention mechanism (multi-head)
    Attention,
    /// Element-wise multiplication
    Multiply,
}

impl Op {
    /// Check if this operation can be fused with another
    pub fn can_fuse_with(&self, next: &Op) -> bool {
        use Op::*;
        matches!(
            (self, next),
            (MatMul, ReLU) | (MatMul, GELU) | (Add, ReLU) | (Add, GELU)
        )
    }

    /// Get the fused operation for two consecutive ops
    pub fn fuse(&self, next: &Op) -> Option<Op> {
        use Op::*;
        match (self, next) {
            (MatMul | Add, ReLU) => Some(FusedLinearReLU),
            (MatMul | Add, GELU) => Some(FusedLinearGELU),
            _ => None,
        }
    }
}

/// A node in the computational graph
#[derive(Debug, Clone)]
pub struct Node {
    /// Unique identifier for this node
    pub id: usize,
    /// Operation to perform
    pub op: Op,
    /// Indices of input tensors in the arena
    pub inputs: Vec<usize>,
    /// Index where the output tensor is stored
    pub output: usize,
    /// Operation-specific metadata (e.g., attention heads, normalization epsilon)
    pub metadata: NodeMetadata,
}

/// Metadata for operation-specific parameters
#[derive(Debug, Clone)]
pub enum NodeMetadata {
    None,
    MatMul {
        m: usize,
        k: usize,
        n: usize,
    },
    Attention {
        num_heads: usize,
        head_dim: usize,
        seq_len: usize,
    },
    LayerNorm {
        epsilon: f32,
    },
}

impl Node {
    /// Check if this node can be fused with the next node
    pub fn is_fusable(&self) -> bool {
        matches!(self.op, Op::MatMul | Op::Add)
    }
}

/// The computational graph representing the entire model
pub struct Graph {
    /// All nodes in topologically sorted order
    nodes: Vec<Node>,
    /// Mapping from tensor index to node that produces it
    tensor_producers: Vec<Option<usize>>,
    /// Total number of tensors in the graph
    num_tensors: usize,
}

impl Graph {
    /// Create a new empty graph
    pub fn new(num_tensors: usize) -> Self {
        Self {
            nodes: Vec::new(),
            tensor_producers: vec![None; num_tensors],
            num_tensors,
        }
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, node: Node) -> usize {
        let node_id = self.nodes.len();
        
        // Register this node as the producer of its output tensor
        if node.output < self.tensor_producers.len() {
            self.tensor_producers[node.output] = Some(node_id);
        }
        
        self.nodes.push(node);
        node_id
    }

    /// Get a node by ID
    #[inline]
    pub fn get_node(&self, id: usize) -> Option<&Node> {
        self.nodes.get(id)
    }

    /// Get the next node in execution order
    #[inline]
    pub fn get_next(&self, current_node_id: usize) -> Option<&Node> {
        self.nodes.get(current_node_id + 1)
    }

    /// Get all nodes in execution order
    #[inline]
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    /// Optimize the graph by fusing compatible operations
    pub fn optimize(&mut self) {
        let mut optimized = Vec::new();
        let mut i = 0;

        while i < self.nodes.len() {
            let current = &self.nodes[i];
            
            // Rust 1.88 let_chains feature
            if let Some(next) = self.nodes.get(i + 1)
                && current.op.can_fuse_with(&next.op)
                && current.is_fusable()
                && current.output == next.inputs[0]  // Ensure they're connected
            {
                // Fuse the operations
                if let Some(fused_op) = current.op.fuse(&next.op) {
                    let fused_node = Node {
                        id: current.id,
                        op: fused_op,
                        inputs: current.inputs.clone(),
                        output: next.output,
                        metadata: current.metadata.clone(),
                    };
                    optimized.push(fused_node);
                    i += 2; // Skip the next node as it's been fused
                    continue;
                }
            }

            optimized.push(current.clone());
            i += 1;
        }

        self.nodes = optimized;
    }

    /// Get graph statistics for debugging
    pub fn stats(&self) -> GraphStats {
        let mut op_counts = std::collections::HashMap::new();
        
        for node in &self.nodes {
            *op_counts.entry(node.op).or_insert(0) += 1;
        }

        GraphStats {
            num_nodes: self.nodes.len(),
            num_tensors: self.num_tensors,
            op_counts,
        }
    }
}

/// Statistics about the graph
pub struct GraphStats {
    pub num_nodes: usize,
    pub num_tensors: usize,
    pub op_counts: std::collections::HashMap<Op, usize>,
}

impl fmt::Display for GraphStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Graph Statistics:")?;
        writeln!(f, "  Nodes: {}", self.num_nodes)?;
        writeln!(f, "  Tensors: {}", self.num_tensors)?;
        writeln!(f, "  Operations:")?;
        for (op, count) in &self.op_counts {
            writeln!(f, "    {:?}: {}", op, count)?;
        }
        Ok(())
    }
}

/// Builder for constructing graphs programmatically
pub struct GraphBuilder {
    graph: Graph,
    next_node_id: usize,
}

impl GraphBuilder {
    pub fn new(num_tensors: usize) -> Self {
        Self {
            graph: Graph::new(num_tensors),
            next_node_id: 0,
        }
    }

    pub fn add_matmul(&mut self, input: usize, weights: usize, output: usize, m: usize, k: usize, n: usize) -> &mut Self {
        let node = Node {
            id: self.next_node_id,
            op: Op::MatMul,
            inputs: vec![input, weights],
            output,
            metadata: NodeMetadata::MatMul { m, k, n },
        };
        self.next_node_id += 1;
        self.graph.add_node(node);
        self
    }

    pub fn add_relu(&mut self, input: usize, output: usize) -> &mut Self {
        let node = Node {
            id: self.next_node_id,
            op: Op::ReLU,
            inputs: vec![input],
            output,
            metadata: NodeMetadata::None,
        };
        self.next_node_id += 1;
        self.graph.add_node(node);
        self
    }

    pub fn add_layer_norm(&mut self, input: usize, output: usize, epsilon: f32) -> &mut Self {
        let node = Node {
            id: self.next_node_id,
            op: Op::LayerNorm,
            inputs: vec![input],
            output,
            metadata: NodeMetadata::LayerNorm { epsilon },
        };
        self.next_node_id += 1;
        self.graph.add_node(node);
        self
    }

    pub fn optimize(mut self) -> Graph {
        self.graph.optimize();
        self.graph
    }

    pub fn build(self) -> Graph {
        self.graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_op_fusion() {
        assert!(Op::MatMul.can_fuse_with(&Op::ReLU));
        assert!(Op::MatMul.can_fuse_with(&Op::GELU));
        assert!(!Op::ReLU.can_fuse_with(&Op::MatMul));
        
        assert_eq!(Op::MatMul.fuse(&Op::ReLU), Some(Op::FusedLinearReLU));
    }

    #[test]
    fn test_graph_builder() {
        let graph = GraphBuilder::new(5)
            .add_matmul(0, 1, 2, 128, 512, 512)
            .add_relu(2, 3)
            .build();

        assert_eq!(graph.nodes().len(), 2);
        assert_eq!(graph.nodes()[0].op, Op::MatMul);
        assert_eq!(graph.nodes()[1].op, Op::ReLU);
    }

    #[test]
    fn test_graph_optimization() {
        let mut graph = GraphBuilder::new(5)
            .add_matmul(0, 1, 2, 128, 512, 512)
            .add_relu(2, 3)
            .build();

        graph.optimize();

        // Should be fused into a single FusedLinearReLU
        assert_eq!(graph.nodes().len(), 1);
        assert_eq!(graph.nodes()[0].op, Op::FusedLinearReLU);
    }
}
