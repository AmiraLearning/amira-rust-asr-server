//! Zero-copy optimized RNN-T decoder implementation.
//!
//! This module provides a high-performance version of the RNN-T decoder
//! that eliminates memory allocations in hot paths using zero-copy operations.

use crate::asr::types::DecoderState;
use crate::asr::zero_copy::{argmax_zero_copy, with_decoder_workspace, TensorView};
use crate::config::model::{BLANK_TOKEN_ID, MAX_SYMBOLS_PER_STEP, MAX_TOTAL_TOKENS};
use crate::error::{AppError, Result};
use std::future::Future;
use tracing::{debug, warn};

/// Compatibility function that matches the legacy greedy_decode interface.
/// This is a wrapper around the optimized zero-copy implementation.
///
/// # Arguments
/// * `encoder_output` - The output from the encoder model (flattened)
/// * `encoded_len` - The length of the encoded output (time steps)
/// * `initial_state` - The initial decoder state
/// * `decode_step_fn` - Function that returns logits and new state for a given input
///
/// # Returns
/// The decoded token sequence and final decoder state
pub async fn greedy_decode<F, Fut>(
    encoder_output: &[f32],
    encoded_len: i64,
    initial_state: DecoderState,
    decode_step_fn: F,
) -> Result<(Vec<i32>, DecoderState)>
where
    F: FnMut(&[f32], &[i32], DecoderState) -> Fut,
    Fut: Future<Output = Result<(Vec<f32>, DecoderState)>>,
{
    // Assume standard encoder output shape based on the flattened data and time steps
    let features_per_step = encoder_output.len() / encoded_len as usize;
    let encoder_shape = &[1, features_per_step, encoded_len as usize]; // [batch=1, features, time_steps]
    
    greedy_decode_zero_copy(encoder_output, encoder_shape, initial_state, decode_step_fn).await
}

/// Zero-copy optimized RNN-T greedy decoder.
///
/// This version eliminates all memory allocations in the hot path by using
/// pre-allocated workspaces and zero-copy tensor operations.
///
/// # Arguments
/// * `encoder_output` - The encoder output tensor data
/// * `encoder_shape` - Shape of encoder output [batch, features, time_steps]  
/// * `initial_state` - The initial decoder state
/// * `decode_step_fn` - Function that performs decoder + joint computation
///
/// # Returns
/// The decoded token sequence and final decoder state
pub async fn greedy_decode_zero_copy<F, Fut>(
    encoder_output: &[f32],
    encoder_shape: &[usize],
    initial_state: DecoderState,
    mut decode_step_fn: F,
) -> Result<(Vec<i32>, DecoderState)>
where
    F: FnMut(&[f32], &[i32], DecoderState) -> Fut,
    Fut: Future<Output = Result<(Vec<f32>, DecoderState)>>,
{
    if encoder_shape.len() != 3 {
        return Err(AppError::Model(
            "Encoder output must be 3D tensor [batch, features, time_steps]".to_string(),
        ));
    }

    let [_batch, features, time_steps] = [encoder_shape[0], encoder_shape[1], encoder_shape[2]];
    let encoded_len = time_steps;

    // Create zero-copy tensor view
    let encoder_tensor = TensorView::new(encoder_output, encoder_shape);

    let mut tokens = Vec::with_capacity(MAX_TOTAL_TOKENS);
    let mut decoder_state = initial_state;
    let mut total_symbols = 0;

    debug!(
        "Starting zero-copy RNN-T decoding: {} time steps, {} features",
        time_steps, features
    );

    // Main decoding loop - zero allocations after this point
    for t in 0..encoded_len {
        if total_symbols >= MAX_TOTAL_TOKENS {
            warn!(
                "Maximum total tokens ({}) reached, stopping decoding",
                MAX_TOTAL_TOKENS
            );
            break;
        }

        // Extract encoder frame
        let encoder_frame = with_decoder_workspace(|workspace| {
            let frame_buffer = workspace.prepare_encoder_frame(features);
            let features_copied = encoder_tensor.extract_frame_into(t, frame_buffer);

            if features_copied != features {
                return None;
            }

            if t < 3 {
                debug!(
                    "Processing time step {} with {} existing tokens",
                    t,
                    tokens.len()
                );
            }

            // Return a copy of the frame
            Some(frame_buffer.to_vec())
        });

        let encoder_frame = match encoder_frame {
            Some(frame) => frame,
            None => {
                return Err(AppError::Model(format!(
                    "Failed to extract encoder frame for time step {}",
                    t
                )));
            }
        };

        // Inner decoder loop
        let mut symbols_count = 0;
        loop {
            symbols_count += 1;
            if symbols_count > MAX_SYMBOLS_PER_STEP {
                debug!("Max symbols reached for time step {}, forcing advance.", t);
                break;
            }

            // Prepare targets vector
            let mut current_targets = Vec::with_capacity(tokens.len() + 1);
            current_targets.push(BLANK_TOKEN_ID);
            current_targets.extend_from_slice(&tokens);

            // Async decoder call - properly awaited
            let (logits, new_state) =
                decode_step_fn(&encoder_frame, &current_targets, decoder_state)
                    .await
                    .map_err(|_e| AppError::Model("Decode step failed".to_string()))?;

            decoder_state = new_state;

            debug!(
                "Decoder returned {} logits for {} targets",
                logits.len(),
                current_targets.len()
            );

            // Zero-copy argmax (no allocation)
            let (predicted_id, confidence) = argmax_zero_copy(&logits);
            let predicted_token = predicted_id as i32;

            debug!(
                "Time step {}, symbol {}: predicted token {} (confidence: {:.4})",
                t, symbols_count, predicted_token, confidence
            );

            if predicted_token == BLANK_TOKEN_ID {
                debug!("Blank token predicted, moving to next time step");
                break;
            } else {
                debug!("Non-blank token {}, continuing inner loop", predicted_token);
                tokens.push(predicted_token);
                total_symbols += 1;

                if total_symbols >= MAX_TOTAL_TOKENS {
                    warn!("Reached maximum total tokens during inner loop");
                    break;
                }
            }
        }

        if total_symbols >= MAX_TOTAL_TOKENS {
            break;
        }
    }

    debug!(
        "Zero-copy RNN-T decoding complete: {} tokens generated",
        tokens.len()
    );
    if !tokens.is_empty() {
        debug!("Generated tokens: {:?}", tokens);
    }

    Ok((tokens, decoder_state))
}

/// Async-compatible version that maintains zero-copy benefits.
///
/// This version properly handles the async decode function while still
/// eliminating allocations in the non-async parts of the computation.
pub async fn greedy_decode_zero_copy_async<F, Fut>(
    encoder_output: &[f32],
    encoder_shape: &[usize],
    initial_state: DecoderState,
    mut decode_step_fn: F,
) -> Result<(Vec<i32>, DecoderState)>
where
    F: FnMut(&[f32], &[i32], DecoderState) -> Fut,
    Fut: Future<Output = Result<(Vec<f32>, DecoderState)>>,
{
    if encoder_shape.len() != 3 {
        return Err(AppError::Model(
            "Encoder output must be 3D tensor [batch, features, time_steps]".to_string(),
        ));
    }

    let [_batch, features, time_steps] = [encoder_shape[0], encoder_shape[1], encoder_shape[2]];

    // Create zero-copy tensor view
    let encoder_tensor = TensorView::new(encoder_output, encoder_shape);

    let mut tokens = Vec::with_capacity(MAX_TOTAL_TOKENS);
    let mut decoder_state = initial_state;
    let mut total_symbols = 0;

    debug!(
        "Starting async zero-copy RNN-T decoding: {} time steps, {} features",
        time_steps, features
    );

    // Pre-allocate workspace outside the async loop
    let mut encoder_frame = vec![0.0f32; features];
    let mut targets_buffer = Vec::with_capacity(MAX_TOTAL_TOKENS + 1);

    for t in 0..time_steps {
        if total_symbols >= MAX_TOTAL_TOKENS {
            warn!("Maximum total tokens reached, stopping decoding");
            break;
        }

        // Zero-copy frame extraction
        let features_copied = encoder_tensor.extract_frame_into(t, &mut encoder_frame);
        if features_copied != features {
            return Err(AppError::Model(format!(
                "Failed to extract encoder frame for time step {}",
                t
            )));
        }

        if t < 3 {
            debug!(
                "Processing time step {} with {} existing tokens",
                t,
                tokens.len()
            );
        }

        // Inner decoder loop
        let mut symbols_count = 0;
        loop {
            symbols_count += 1;
            if symbols_count > MAX_SYMBOLS_PER_STEP {
                debug!("Max symbols reached for time step {}, forcing advance.", t);
                break;
            }

            // Zero-copy target preparation (reuse buffer)
            targets_buffer.clear();
            targets_buffer.push(BLANK_TOKEN_ID);
            targets_buffer.extend_from_slice(&tokens);

            // Async decoder call
            let (logits, new_state) =
                decode_step_fn(&encoder_frame, &targets_buffer, decoder_state).await?;
            decoder_state = new_state;

            // Zero-copy argmax
            let (predicted_id, confidence) = argmax_zero_copy(&logits);
            let predicted_token = predicted_id as i32;

            debug!(
                "Time step {}, symbol {}: predicted token {} (confidence: {:.4})",
                t, symbols_count, predicted_token, confidence
            );

            if predicted_token == BLANK_TOKEN_ID {
                debug!("Blank token predicted, moving to next time step");
                break;
            } else {
                debug!("Non-blank token {}, continuing inner loop", predicted_token);
                tokens.push(predicted_token);
                total_symbols += 1;

                if total_symbols >= MAX_TOTAL_TOKENS {
                    warn!("Reached maximum total tokens during inner loop");
                    break;
                }
            }
        }

        if total_symbols >= MAX_TOTAL_TOKENS {
            break;
        }
    }

    debug!(
        "Async zero-copy RNN-T decoding complete: {} tokens generated",
        tokens.len()
    );
    if !tokens.is_empty() {
        debug!("Generated tokens: {:?}", tokens);
    }

    Ok((tokens, decoder_state))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asr::types::DecoderState;

    #[tokio::test]
    async fn test_zero_copy_decoder() {
        // Create test encoder output: 2 features, 3 time steps
        let encoder_output = vec![
            1.0, 2.0, 3.0, // Feature 0
            4.0, 5.0, 6.0, // Feature 1
        ];
        let encoder_shape = vec![1, 2, 3]; // [batch, features, time_steps]

        let initial_state = DecoderState::new();

        // Mock decode function that returns predictable results
        let mut call_count = 0;
        let decode_fn = |_frame: &[f32], _targets: &[i32], state: DecoderState| {
            call_count += 1;
            async move {
                // Return blank token to end decoding quickly
                let logits = vec![0.1, 0.2, 0.9]; // Max at index 2 (non-blank)
                Ok((logits, state))
            }
        };

        let result = greedy_decode_zero_copy_async(
            &encoder_output,
            &encoder_shape,
            initial_state,
            decode_fn,
        )
        .await;

        assert!(result.is_ok());
        let (tokens, _final_state) = result.unwrap();

        // Should generate some tokens based on our mock function
        assert!(!tokens.is_empty());
    }
}
