//! RNN-T greedy decoder algorithm.
//!
//! This module provides the core RNN-T greedy search algorithm,
//! which is the primary decoding method for the ASR pipeline.

use crate::asr::types::DecoderState;
use crate::config::model::{BLANK_TOKEN_ID, MAX_SYMBOLS_PER_STEP, MAX_TOTAL_TOKENS};
use crate::error::{AppError, Result};
use std::future::Future;
// use tracing::{debug, warn};  // Temporarily disabled
// Temporary no-op macros to replace tracing
macro_rules! debug { ($($tt:tt)*) => {}; }
macro_rules! warn { ($($tt:tt)*) => {}; }

/// Performs a greedy RNN-T search.
///
/// This function contains the core decoding loop, decoupled from Triton
/// by accepting an `async` closure (`decode_step_fn`) that provides the
/// logits for a given state, making the algorithm itself pure and testable.
///
/// # Arguments
/// * `encoder_output` - The output from the encoder model
/// * `encoded_len` - The length of the encoded output
/// * `initial_state` - The initial decoder state
/// * `decode_step_fn` - A function that returns logits and new state for a given input
///
/// # Returns
/// The decoded token IDs and the final decoder state
pub async fn greedy_decode<F, Fut>(
    encoder_output: &[f32],
    encoded_len: i64,
    initial_state: DecoderState,
    mut decode_step_fn: F,
) -> Result<(Vec<i32>, DecoderState)>
where
    F: FnMut(&[f32], &[i32], DecoderState) -> Fut,
    Fut: Future<Output = Result<(Vec<f32>, DecoderState)>>,
{
    let mut tokens: Vec<i32> = Vec::new();
    let mut decoder_state = initial_state;

    debug!(
        "Starting RNN-T greedy search for {} time steps",
        encoded_len
    );

    // 1. Outer loop: Iterate through each frame from the ENCODER (acoustic model)
    for t in 0..encoded_len {
        if tokens.len() >= MAX_TOTAL_TOKENS {
            warn!(
                "Reached maximum token limit of {}, stopping decode",
                MAX_TOTAL_TOKENS
            );
            break;
        }

        // The encoder output tensor has shape [1, 1024, encoded_len].
        // In a flattened Vec, this is a row-major layout, meaning it's 1024 contiguous blocks,
        // each of size encoded_len.
        // To get the acoustic vector for a single time step `t`, we need to pick the
        // t-th element from each of the 1024 feature rows.
        let feature_dim = 1024;
        let mut encoder_frame_vec = Vec::with_capacity(feature_dim);
        for i in 0..feature_dim {
            // This is the formula to access the element at (row i, column t)
            // in a row-major matrix.
            let index = i * (encoded_len as usize) + (t as usize);

            if index >= encoder_output.len() {
                return Err(AppError::Model(format!(
                    "Index out of bounds when creating encoder frame for time step {}. This likely indicates a mismatch between 'encoded_len' and the actual tensor dimensions.",
                    t
                )));
            }
            encoder_frame_vec.push(encoder_output[index]);
        }
        let encoder_frame = &encoder_frame_vec; // Use a reference for the subsequent call

        if t < 3 {
            debug!(
                "Processing time step {} with {} existing tokens",
                t,
                tokens.len()
            );
        }

        // 2. Inner loop: Iterate through the DECODER (prediction network)
        //    until it emits a BLANK token.
        let mut symbol_count_for_step = 0;
        loop {
            symbol_count_for_step += 1;
            if symbol_count_for_step > MAX_SYMBOLS_PER_STEP {
                debug!("Max symbols reached for time step {}, forcing advance.", t);
                break;
            }

            // To predict the next token, the decoder needs the full sequence history,
            // which starts with the blank token (acting as a start-of-sequence token).
            let mut current_targets = vec![BLANK_TOKEN_ID];
            current_targets.extend_from_slice(&tokens);

            // 3. Call the joint network. It takes the acoustic frame (which is fixed for this `t`)
            //    and the current token hypothesis to produce logits and a NEW decoder state.
            let (logits, new_state) =
                decode_step_fn(encoder_frame, &current_targets, decoder_state).await?;

            debug!(
                "Decoder returned {} logits for {} targets",
                logits.len(),
                current_targets.len()
            );

            // Extract the logits for the next token
            let (last_token_logits, _) = extract_last_token_logits(&logits)?;

            // Log top predictions for debugging (for first few and last time steps)
            if t < 3 || (t == encoded_len - 1 && symbol_count_for_step <= 3) {
                log_top_predictions(t, symbol_count_for_step, last_token_logits);
            }

            // 4. Find the most likely token (argmax) from these logits.
            let (predicted_token, best_score) = last_token_logits.iter().enumerate().fold(
                (0, f32::NEG_INFINITY),
                |(idx_max, val_max), (idx, &val)| {
                    if val > val_max {
                        (idx as i32, val)
                    } else {
                        (idx_max, val_max)
                    }
                },
            );

            if t < 3 || t == encoded_len - 1 {
                debug!(
                    "Predicted token: {} (score: {:.2})",
                    predicted_token, best_score
                );
                // Also log the blank token score for comparison
                if let Some(&blank_score) = last_token_logits.get(BLANK_TOKEN_ID as usize) {
                    debug!(
                        "Blank token score: {:.2} (token {})",
                        blank_score, BLANK_TOKEN_ID
                    );
                }
            }

            // 5. CRITICAL: Unconditionally update the decoder state. The `new_state`
            //    contains the updated recurrent state of the prediction network.
            decoder_state = new_state;

            // 6. Check the predicted token.
            if predicted_token == BLANK_TOKEN_ID {
                // The model says "I'm done with this audio frame."
                // Break the inner loop to advance to the next audio frame `t+1`.
                if t < 3 || t == encoded_len - 1 {
                    debug!("Blank predicted, advancing to next time step");
                }
                break;
            } else {
                // The model emitted a character.
                // - Add it to our list of tokens.
                // - The inner loop will run again. In the next iteration,
                //   the decoder will be conditioned on the newly updated
                //   `tokens` list and the new `decoder_state`.
                tokens.push(predicted_token);
                if t < 3 || t == encoded_len - 1 {
                    debug!(
                        "Added token {} to sequence, now have {} tokens",
                        predicted_token,
                        tokens.len()
                    );
                }
            }
        }
    }

    debug!("RNN-T decoding complete, generated {} tokens", tokens.len());
    Ok((tokens, decoder_state))
}

/// Extract the logits for the last token position.
///
/// # Arguments
/// * `logits` - The raw logits from the model
///
/// # Returns
/// The slice of logits for the last token position and its size
fn extract_last_token_logits(logits: &[f32]) -> Result<(&[f32], usize)> {
    use crate::config::model::VOCABULARY_SIZE;

    if logits.len() == VOCABULARY_SIZE {
        // Model returned logits for just the next position (ideal case)
        return Ok((logits, VOCABULARY_SIZE));
    }

    // Model returned logits for all target positions
    let num_positions = logits.len() / VOCABULARY_SIZE;

    if num_positions == 0 {
        return Err(AppError::Model(
            "No logits returned from decoder".to_string(),
        ));
    }

    // We want the last position's logits
    let last_position = num_positions - 1;
    let start_idx = last_position * VOCABULARY_SIZE;
    let end_idx = start_idx + VOCABULARY_SIZE;

    if end_idx > logits.len() {
        return Err(AppError::Model(format!(
            "Cannot extract logits: indices {}-{} exceed total length {}",
            start_idx,
            end_idx,
            logits.len()
        )));
    }

    debug!(
        "Extracting logits from position {} (indices {}-{})",
        last_position, start_idx, end_idx
    );
    Ok((&logits[start_idx..end_idx], VOCABULARY_SIZE))
}

/// Log the top N predictions for debugging.
///
/// # Arguments
/// * `t` - The current time step
/// * `symbol_count` - The current symbol count within this time step
/// * `logits` - The logits to analyze
fn log_top_predictions(t: i64, symbol_count: usize, logits: &[f32]) {
    use crate::config::model::BLANK_TOKEN_ID;

    let mut scores: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &s)| (i, s)).collect();

    // Handle NaN values gracefully - treat NaN as less than any other value
    scores.sort_by(|a, b| {
        match b.1.partial_cmp(&a.1) {
            Some(ordering) => ordering,
            None => {
                // If either value is NaN, sort NaN values last
                if b.1.is_nan() && !a.1.is_nan() {
                    std::cmp::Ordering::Less
                } else if !b.1.is_nan() && a.1.is_nan() {
                    std::cmp::Ordering::Greater
                } else {
                    // Both NaN, consider them equal
                    std::cmp::Ordering::Equal
                }
            }
        }
    });

    debug!("Time {}, symbol {}: Top 5 predictions:", t, symbol_count);
    for (i, (token_id, score)) in scores.iter().take(5).enumerate() {
        let token_str = if *token_id == BLANK_TOKEN_ID as usize {
            "[BLANK]".to_string()
        } else {
            format!("{}", token_id)
        };
        debug!("  {}: token {} (score: {:.2})", i + 1, token_str, score);
    }
}
