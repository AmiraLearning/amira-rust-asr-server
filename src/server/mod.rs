//! Web server and API implementation.
//!
//! This module provides the HTTP and WebSocket server functionality
//! for the ASR service.

mod handlers;
// mod io_uring_websocket;  // Temporarily disabled - requires additional io_uring dependencies
mod metrics;
mod state;
mod stream;

pub use handlers::{create_router, health_check, metrics_handler, reset_batch_count};
// pub use io_uring_websocket::{IoUringWebSocketSession, IoUringWebSocketFactory, IoUringConfig, SessionStats};
pub use metrics::ServiceMetrics;
pub use state::AppState;
pub use stream::{StreamProcessor, StreamRequest};
