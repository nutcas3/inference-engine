//! Output Sink
//! 
//! Handles inference results with backpressure support.
//! If the consumer (network, disk, etc.) is slow, the Sink signals
//! the upstream to stop sending more data.

use futures::sink::Sink;
use std::pin::Pin;
use std::task::{Context, Poll};

/// Result of an inference operation
#[derive(Debug, Clone)]
pub struct InferenceResult {
    /// Request ID this result corresponds to
    pub request_id: u64,
    /// Output tensor data
    pub data: Vec<f32>,
    /// Inference latency in microseconds
    pub latency_us: u64,
}

/// Sink for inference results
/// 
/// This can write to various destinations:
/// - WebSocket for streaming responses
/// - gRPC stream for RPC calls
/// - Local buffer for batch processing
pub struct ResultSink {
    /// Internal buffer for results
    buffer: Vec<InferenceResult>,
    /// Maximum buffer size before backpressure
    max_buffer_size: usize,
}

impl ResultSink {
    pub fn new(max_buffer_size: usize) -> Self {
        Self {
            buffer: Vec::new(),
            max_buffer_size,
        }
    }

    /// Check if the sink has capacity
    fn has_capacity(&self) -> bool {
        self.buffer.len() < self.max_buffer_size
    }

    /// Drain the buffer (simulate sending to network/disk)
    pub fn drain(&mut self) -> Vec<InferenceResult> {
        std::mem::take(&mut self.buffer)
    }
}

impl Sink<InferenceResult> for ResultSink {
    type Error = SinkError;

    fn poll_ready(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        if self.has_capacity() {
            Poll::Ready(Ok(()))
        } else {
            // Backpressure: not ready to accept more results
            Poll::Pending
        }
    }

    fn start_send(mut self: Pin<&mut Self>, item: InferenceResult) -> Result<(), Self::Error> {
        if !self.has_capacity() {
            return Err(SinkError::BufferFull);
        }
        
        self.buffer.push(item);
        Ok(())
    }

    fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        // In production, this would flush to network/disk
        Poll::Ready(Ok(()))
    }

    fn poll_close(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SinkError {
    #[error("Sink buffer is full")]
    BufferFull,
    
    #[error("IO error: {0}")]
    Io(String),
}

/// WebSocket sink (placeholder - requires tokio-tungstenite in production)
pub struct WebSocketSink {
    _endpoint: String,
}

impl WebSocketSink {
    pub fn new(endpoint: String) -> Self {
        Self {
            _endpoint: endpoint,
        }
    }
}

impl Sink<InferenceResult> for WebSocketSink {
    type Error = SinkError;

    fn poll_ready(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        // Check if WebSocket is writable
        Poll::Ready(Ok(()))
    }

    fn start_send(self: Pin<&mut Self>, item: InferenceResult) -> Result<(), Self::Error> {
        // Serialize and send via WebSocket
        tracing::debug!("Sending result for request {}", item.request_id);
        Ok(())
    }

    fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn poll_close(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        // Close WebSocket connection
        Poll::Ready(Ok(()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::SinkExt;

    #[test]
    fn test_result_sink_backpressure() {
        let mut sink = ResultSink::new(2);
        
        futures::executor::block_on(async {
            // Can send up to capacity
            let result1 = InferenceResult {
                request_id: 1,
                data: vec![1.0, 2.0],
                latency_us: 100,
            };
            sink.send(result1).await.expect("Should send");
            
            let result2 = InferenceResult {
                request_id: 2,
                data: vec![3.0, 4.0],
                latency_us: 150,
            };
            sink.send(result2).await.expect("Should send");
            
            // Drain to free capacity
            let drained = sink.drain();
            assert_eq!(drained.len(), 2);
            
            // Can send again
            let result3 = InferenceResult {
                request_id: 3,
                data: vec![5.0, 6.0],
                latency_us: 120,
            };
            sink.send(result3).await.expect("Should send after drain");
        });
    }
}
