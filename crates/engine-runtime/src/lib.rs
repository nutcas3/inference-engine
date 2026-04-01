//! Engine Runtime
//! 
//! The "Control Plane" that bridges async I/O with synchronous inference.
//! Provides Stream/Sink abstractions without forcing the core to be async.

pub mod stream;
pub mod sink;

use engine_core::{InferenceEngine, arena::Section};
use stream::{InferenceRequest, RequestStream, WorkspaceManager};
use sink::{InferenceResult, ResultSink};
use futures::{StreamExt, SinkExt};
use std::time::Instant;

/// Hybrid runtime: combines async I/O with synchronous computation
pub struct HybridRuntime {
    engine: InferenceEngine,
    workspace_manager: WorkspaceManager,
}

impl HybridRuntime {
    pub fn new(engine: InferenceEngine, num_concurrent_requests: usize) -> Self {
        Self {
            engine,
            workspace_manager: WorkspaceManager::new(num_concurrent_requests),
        }
    }

    /// Process a stream of requests and send results to a sink
    /// 
    /// This is the main execution loop that ties everything together:
    /// 1. Stream provides requests
    /// 2. Workspace manager provides memory slots
    /// 3. Engine performs computation
    /// 4. Sink receives results (with backpressure)
    pub async fn run<S>(
        &mut self,
        mut request_stream: RequestStream,
        mut result_sink: S,
    ) -> Result<(), RuntimeError>
    where
        S: futures::Sink<InferenceResult> + Unpin,
        S::Error: std::fmt::Debug,
    {
        while let Some(request) = request_stream.next().await {
            tracing::info!("Processing request {}", request.id);

            // Wait for workspace slot (backpressure on input)
            let slot = loop {
                if let Some(s) = self.workspace_manager.try_acquire() {
                    break s;
                }
                // In production, would yield to executor here
                // For now, just continue
            };

            // Copy input data to IO buffer
            {
                let io_section = self.engine.arena_mut().section_slice_mut(Section::IO);
                let copy_len = request.data.len().min(io_section.len());
                io_section[..copy_len].copy_from_slice(&request.data[..copy_len]);
            }

            // Execute inference (synchronous hot path)
            let start = Instant::now();
            self.engine.execute()
                .map_err(|e| RuntimeError::ExecutionFailed(e.to_string()))?;
            let latency = start.elapsed();

            // Read output from IO buffer
            let output_data = {
                let io_section = self.engine.arena().section_slice(Section::IO);
                // In production, would know exact output size
                io_section[..100].to_vec()
            };

            // Release workspace
            self.workspace_manager.release(slot);

            // Send result to sink (backpressure on output)
            let result = InferenceResult {
                request_id: request.id,
                data: output_data,
                latency_us: latency.as_micros() as u64,
            };

            result_sink.send(result).await
                .map_err(|e| RuntimeError::SinkFailed(format!("{:?}", e)))?;

            tracing::info!("Completed request {} in {:?}", request.id, latency);
        }

        Ok(())
    }

    /// Synchronous single-shot inference (for testing)
    pub fn infer_sync(&mut self, input: &[f32]) -> Result<Vec<f32>, RuntimeError> {
        // Copy input
        {
            let io_section = self.engine.arena_mut().section_slice_mut(Section::IO);
            let copy_len = input.len().min(io_section.len());
            io_section[..copy_len].copy_from_slice(&input[..copy_len]);
        }

        // Execute
        self.engine.execute()
            .map_err(|e| RuntimeError::ExecutionFailed(e.to_string()))?;

        // Copy output
        let output = {
            let io_section = self.engine.arena().section_slice(Section::IO);
            io_section[..100].to_vec()
        };

        Ok(output)
    }

    /// Get reference to the underlying engine
    pub fn engine(&self) -> &InferenceEngine {
        &self.engine
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error("Execution failed: {0}")]
    ExecutionFailed(String),
    
    #[error("Sink failed: {0}")]
    SinkFailed(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_core::graph::GraphBuilder;

    #[test]
    fn test_sync_inference() {
        // Build simple graph
        let graph = GraphBuilder::new(10)
            .add_matmul(0, 1, 2, 2, 3, 2)
            .add_relu(2, 3)
            .build();

        let engine = engine_core::InferenceEngine::new(graph, 100, 200, 500);
        let mut runtime = HybridRuntime::new(engine, 4);

        // Run synchronous inference
        let input = vec![1.0, 2.0, 3.0];
        let result = runtime.infer_sync(&input);
        assert!(result.is_ok());
    }

    #[test]
    fn test_async_pipeline() {
        let graph = GraphBuilder::new(10)
            .add_matmul(0, 1, 2, 2, 3, 2)
            .build();

        let engine = engine_core::InferenceEngine::new(graph, 100, 200, 500);
        let mut runtime = HybridRuntime::new(engine, 4);

        // Create test requests
        let requests = vec![
            InferenceRequest {
                id: 1,
                data: vec![1.0; 10],
                batch_size: 1,
            },
            InferenceRequest {
                id: 2,
                data: vec![2.0; 10],
                batch_size: 1,
            },
        ];

        let stream = RequestStream::new(requests);
        let sink = ResultSink::new(10);

        // Run async pipeline
        futures::executor::block_on(async {
            let result = runtime.run(stream, sink).await;
            assert!(result.is_ok());
        });
    }
}
