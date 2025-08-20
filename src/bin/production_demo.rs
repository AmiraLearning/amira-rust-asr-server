//! Production Reliability Features Demo
//!
//! This binary demonstrates the production reliability features:
//! - Circuit breaker pattern for fault tolerance
//! - Graceful shutdown handling
//! - Basic health checks and monitoring

use amira_rust_asr_server::reliability::{
    circuit_breaker::{CircuitBreaker, CircuitBreakerConfig},
    graceful_shutdown::GracefulShutdown,
};

use axum::{extract::State, http::StatusCode, response::Json, routing::get, Router};
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;

use tracing::{error, info, warn};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    info!("🚀 Starting Production Reliability Features Demo");

    // Create circuit breaker with aggressive settings for demo
    let circuit_config = CircuitBreakerConfig {
        failure_threshold: 3,
        recovery_timeout: Duration::from_secs(10),
        request_timeout: Duration::from_secs(2),
        success_threshold: 2,
        window_size: Duration::from_secs(30),
    };
    let circuit_breaker = Arc::new(CircuitBreaker::new(circuit_config));

    // Initialize graceful shutdown
    let shutdown_handler = GracefulShutdown::new();
    shutdown_handler.wait_for_signal().await;

    // Create demo routes
    let app = Router::new()
        .route("/health", get(health_check))
        .route("/demo/success", get(demo_success))
        .route("/demo/failure", get(demo_failure))
        .route("/demo/circuit-status", get(circuit_status))
        .with_state(AppState {
            circuit_breaker: Arc::clone(&circuit_breaker),
        });

    info!("🌐 Starting demo server on http://localhost:3000");
    info!("🔧 Demo endpoints:");
    info!("   GET /health - Health check");
    info!("   GET /demo/success - Always succeeds");
    info!("   GET /demo/failure - Always fails (to test circuit breaker)");
    info!("   GET /demo/circuit-status - Circuit breaker status");
    info!("💡 Try calling /demo/failure multiple times to see circuit breaker open!");
    info!("🛑 Press Ctrl+C to test graceful shutdown");

    // Start demo scenarios in background
    tokio::spawn(demo_scenarios(Arc::clone(&circuit_breaker)));

    // Start the server
    let listener = tokio::net::TcpListener::bind("127.0.0.1:3000").await?;

    let server = axum::serve(listener, app);

    // Run server with graceful shutdown
    let graceful = server.with_graceful_shutdown(async move {
        let mut shutdown_guard = shutdown_handler.subscribe();
        let _ = shutdown_guard.recv().await;
        info!("🛑 Graceful shutdown initiated");
    });

    if let Err(e) = graceful.await {
        error!("Server error: {}", e);
    }

    info!("✅ Server shut down gracefully");
    Ok(())
}

#[derive(Clone)]
struct AppState {
    circuit_breaker: Arc<CircuitBreaker>,
}

async fn health_check() -> Json<Value> {
    Json(json!({
        "status": "healthy",
        "service": "amira-asr-production-demo",
        "features": [
            "circuit_breaker",
            "graceful_shutdown"
        ]
    }))
}

async fn demo_success(State(state): State<AppState>) -> Json<Value> {
    // Simulate successful operation through circuit breaker
    let result = state
        .circuit_breaker
        .call(async {
            sleep(Duration::from_millis(100)).await;
            Ok::<String, std::io::Error>("Success!".to_string())
        })
        .await;

    match result {
        Ok(message) => Json(json!({
            "status": "success",
            "message": message,
            "circuit_state": format!("{:?}", state.circuit_breaker.state()),
        })),
        Err(e) => Json(json!({
            "status": "error",
            "error": e.to_string(),
            "circuit_state": format!("{:?}", state.circuit_breaker.state()),
        })),
    }
}

async fn demo_failure(State(state): State<AppState>) -> (StatusCode, Json<Value>) {
    // Simulate failing operation through circuit breaker
    let result = state
        .circuit_breaker
        .call(async {
            sleep(Duration::from_millis(50)).await;
            Err::<String, std::io::Error>(std::io::Error::other(
                "Simulated failure for demo",
            ))
        })
        .await;

    let response = match result {
        Ok(_) => unreachable!(),
        Err(e) => {
            json!({
                "status": "error",
                "error": e.to_string(),
                "circuit_state": format!("{:?}", state.circuit_breaker.state()),
                "tip": "Call this endpoint multiple times to see circuit breaker open"
            })
        }
    };

    (StatusCode::INTERNAL_SERVER_ERROR, Json(response))
}

async fn circuit_status(State(state): State<AppState>) -> Json<Value> {
    let metrics = state.circuit_breaker.metrics();

    Json(json!({
        "circuit_breaker": {
            "state": format!("{:?}", state.circuit_breaker.state()),
            "metrics": {
                "total_requests": metrics.total_requests.load(std::sync::atomic::Ordering::Relaxed),
                "successful_requests": metrics.successful_requests.load(std::sync::atomic::Ordering::Relaxed),
                "failed_requests": metrics.failed_requests.load(std::sync::atomic::Ordering::Relaxed),
                "rejected_requests": metrics.rejected_requests.load(std::sync::atomic::Ordering::Relaxed),
                "circuit_opens": metrics.circuit_opens.load(std::sync::atomic::Ordering::Relaxed),
                "circuit_closes": metrics.circuit_closes.load(std::sync::atomic::Ordering::Relaxed),
                "failure_rate_percent": format!("{:.1}%", metrics.failure_rate()),
                "success_rate_percent": format!("{:.1}%", metrics.success_rate()),
            }
        }
    }))
}

async fn demo_scenarios(circuit_breaker: Arc<CircuitBreaker>) {
    info!("🎬 Starting demo scenarios");

    // Wait a bit before starting scenarios
    sleep(Duration::from_secs(5)).await;

    // Scenario 1: Test successful operations
    info!("📋 Scenario 1: Testing successful operations");
    for i in 1..=3 {
        let result = circuit_breaker
            .call(async {
                sleep(Duration::from_millis(100)).await;
                Ok::<String, std::io::Error>(format!("Success {}", i))
            })
            .await;

        info!("✅ Operation {}: {:?}", i, result);
        sleep(Duration::from_millis(500)).await;
    }

    sleep(Duration::from_secs(2)).await;

    // Scenario 2: Cause circuit breaker to open
    info!("📋 Scenario 2: Causing circuit breaker to open with failures");
    for i in 1..=5 {
        let result = circuit_breaker
            .call(async {
                sleep(Duration::from_millis(50)).await;
                Err::<String, std::io::Error>(std::io::Error::other(
                    format!("Demo failure {}", i),
                ))
            })
            .await;

        match result {
            Ok(_) => info!("✅ Operation {}: Success", i),
            Err(e) => warn!("❌ Operation {}: {}", i, e),
        }

        info!("🔵 Circuit state: {:?}", circuit_breaker.state());
        sleep(Duration::from_millis(500)).await;
    }

    // Scenario 3: Wait for recovery and test
    info!("📋 Scenario 3: Waiting for circuit recovery");
    sleep(Duration::from_secs(12)).await;

    info!("📋 Scenario 4: Testing recovery with successful operations");
    for i in 1..=3 {
        let result = circuit_breaker
            .call(async {
                sleep(Duration::from_millis(100)).await;
                Ok::<String, std::io::Error>(format!("Recovery success {}", i))
            })
            .await;

        info!("✅ Recovery operation {}: {:?}", i, result);
        info!("🔵 Circuit state: {:?}", circuit_breaker.state());
        sleep(Duration::from_millis(500)).await;
    }

    info!("🎬 Demo scenarios completed!");
    info!("🔧 Try the demo endpoints manually for interactive testing");
}
