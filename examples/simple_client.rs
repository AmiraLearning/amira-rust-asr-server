// examples/simple_client.rs
//! Simple example client for testing the ASR server

use futures_util::{SinkExt, StreamExt};
use serde_json::json;
use std::time::Duration;
use tokio_tungstenite::{connect_async, tungstenite::Message};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Test batch endpoint
    println!("Testing batch endpoint...");
    test_batch_endpoint().await?;

    // Test streaming endpoint
    println!("Testing streaming endpoint...");
    test_streaming_endpoint().await?;

    Ok(())
}

async fn test_batch_endpoint() -> Result<(), Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();

    // Generate dummy audio data (silence)
    let audio_data: Vec<u8> = vec![0; 16000 * 2]; // 1 second of 16-bit silence

    let request_body = json!({
        "audio_buffer": audio_data,
        "description": "Test audio"
    });

    let response = client
        .post("http://localhost:8057/v2/decode/batch/default")
        .json(&request_body)
        .send()
        .await?;

    if response.status().is_success() {
        let result: serde_json::Value = response.json().await?;
        println!("Batch result: {}", serde_json::to_string_pretty(&result)?);
    } else {
        println!("Batch request failed: {}", response.status());
    }

    Ok(())
}

async fn test_streaming_endpoint() -> Result<(), Box<dyn std::error::Error>> {
    let url = "ws://localhost:8057/v2/decode/stream/default";
    let (ws_stream, _) = connect_async(url).await?;
    let (mut write, mut read) = ws_stream.split();

    // Spawn task to read responses
    let read_task = tokio::spawn(async move {
        while let Some(message) = read.next().await {
            match message {
                Ok(Message::Text(text)) => {
                    println!("Received: {}", text);
                }
                Ok(Message::Close(_)) => {
                    println!("Connection closed");
                    break;
                }
                Err(e) => {
                    println!("Error: {}", e);
                    break;
                }
                _ => {}
            }
        }
    });

    // Send test audio chunks
    for i in 0..5 {
        // Generate dummy audio chunk (silence)
        let chunk: Vec<u8> = vec![0; 1600 * 2]; // 100ms of 16-bit silence

        write.send(Message::Binary(chunk)).await?;
        println!("Sent chunk {}", i + 1);

        // Wait between chunks
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    // Send end-of-stream signal
    write.send(Message::Binary(vec![0x00])).await?;

    // Wait for responses
    tokio::time::sleep(Duration::from_secs(2)).await;

    // Close connection
    write.close().await?;
    read_task.await?;

    Ok(())
}
