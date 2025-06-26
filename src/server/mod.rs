//! Web server and API implementation.
//!
//! This module provides the HTTP and WebSocket server functionality
//! for the ASR service.

mod handlers;
mod metrics;
mod state;
mod stream;

pub use handlers::{create_router, health_check, metrics_handler, reset_batch_count};
pub use metrics::ServiceMetrics;
pub use state::AppState;
pub use stream::{StreamProcessor, StreamRequest};
