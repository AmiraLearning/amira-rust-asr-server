//! Transcript weaving utilities.
//!
//! This module provides functionality for intelligently weaving together
//! transcription segments from overlapping audio chunks, based on the
//! original Python implementation's techniques.

use crate::asr::types::{ALPHA, EXPECTED_SILENCE_RATIO, MAX_ALIGN_DIST};
// use tracing::{debug, info};  // Temporarily disabled
macro_rules! debug { ($($tt:tt)*) => {}; }
macro_rules! info { ($($tt:tt)*) => {}; }

/// Calculate the Levenshtein distance between two strings.
///
/// This is a measure of the difference between two strings,
/// calculated as the minimum number of single-character edits
/// (i.e., insertions, deletions, or substitutions) required to
/// change one string into another.
pub fn levenshtein_distance(s1: &str, s2: &str) -> usize {
    if s1 == s2 {
        return 0;
    }

    if s1.is_empty() {
        return s2.len();
    }
    if s2.is_empty() {
        return s1.len();
    }

    let s1_chars: Vec<char> = s1.chars().collect();
    let s2_chars: Vec<char> = s2.chars().collect();

    let s1_len = s1_chars.len();
    let s2_len = s2_chars.len();

    // Create a matrix of size (s1_len+1) x (s2_len+1)
    let mut matrix = vec![vec![0; s2_len + 1]; s1_len + 1];

    // Initialize the first row and column
    for i in 0..=s1_len {
        matrix[i][0] = i;
    }
    for j in 0..=s2_len {
        matrix[0][j] = j;
    }

    // Fill the matrix
    for i in 1..=s1_len {
        for j in 1..=s2_len {
            let cost = if s1_chars[i - 1] == s2_chars[j - 1] {
                0
            } else {
                1
            };

            matrix[i][j] = std::cmp::min(
                matrix[i - 1][j] + 1, // deletion
                std::cmp::min(
                    matrix[i][j - 1] + 1,        // insertion
                    matrix[i - 1][j - 1] + cost, // substitution
                ),
            );
        }
    }

    matrix[s1_len][s2_len]
}

/// Calculate the normalized word distance between two strings.
///
/// Returns a value between 0 and 1, where 0 means identical strings
/// and 1 means completely different strings.
pub fn word_distance(first: &str, second: &str) -> f32 {
    if first == second {
        return 0.0;
    }

    let first_len = first.len();
    let second_len = second.len();

    if first_len == 0 && second_len == 0 {
        return 0.0;
    }

    let distance = levenshtein_distance(first, second) as f32;
    2.0 * distance / (first_len + second_len) as f32
}

/// Calculate overlap prior based on the lengths of segments and overlap.
///
/// This is a Gaussian-like function that weights the likelihood of an overlap
/// based on the expected overlap size given the segment lengths.
pub fn overlap_prior(first: &str, second: &str, overlap: usize, percent_time: f32) -> f32 {
    let first_len = first.len();
    let second_len = second.len();

    let mu = (first_len as f32 * 3.0 + second_len as f32 * 2.0) * percent_time / 5.0;
    let sigma = mu / 2.0;

    let diff = (overlap as f32 - mu) / sigma;
    let exponent = -0.5 * diff * diff;

    let normalization = sigma * (2.0 * std::f32::consts::PI).sqrt();
    exponent.exp() / normalization
}

/// Calculate the distance score.
///
/// Transforms a distance metric into a similarity score.
pub fn dist_score(dist: f32) -> f32 {
    1.0 / (dist + ALPHA) - 1.0 / (1.0 + ALPHA)
}

/// Calculate the alignment score between two strings with a given overlap.
///
/// This score measures how well the end of the first string aligns
/// with the beginning of the second string.
pub fn align_score(first: &str, second: &str, overlap: usize, percent_time_overlap: f32) -> f32 {
    if first.len() < overlap || second.len() < overlap {
        return 0.0;
    }

    let first_end = match first
        .char_indices()
        .nth_back(first.chars().count().saturating_sub(overlap))
    {
        Some((idx, _)) => &first[idx..],
        None => first,
    };

    let second_start = match second.char_indices().nth(overlap.saturating_sub(1)) {
        Some((idx, _)) => &second[..idx],
        None => second,
    };

    let dist = word_distance(first_end, second_start);

    if dist > MAX_ALIGN_DIST {
        return 0.0;
    }

    overlap_prior(first, second, overlap, percent_time_overlap) * dist_score(dist)
}

/// Calculate the trim alignment score.
///
/// Used to find the optimal way to trim overlapping segments.
pub fn trim_align_score(first: &str, second: &str, overlap: usize) -> f32 {
    if first.is_empty() || second.is_empty() || overlap == 0 {
        return 0.0;
    }

    let first_end = match first
        .char_indices()
        .nth_back(first.chars().count().saturating_sub(overlap))
    {
        Some((idx, _)) => &first[idx..],
        None => first,
    };

    let second_start = match second.char_indices().nth(overlap.saturating_sub(1)) {
        Some((idx, _)) => &second[..idx],
        None => second,
    };

    let dist = word_distance(first_end, second_start);

    if dist > MAX_ALIGN_DIST {
        return 0.0;
    }

    (1.0 - dist) * (overlap as f32).sqrt()
}

/// Find the best alignment between two strings.
///
/// Determines the optimal overlap size and alignment score.
pub fn best_alignment(first: &str, second: &str, percent_time_overlap: f32) -> (usize, f32) {
    let mut best_score = 0.0;
    let mut best_overlap = 0;

    let first_len = first.chars().count();
    let second_len = second.chars().count();

    if first_len == 0 || second_len == 0 {
        return (0, 0.0);
    }

    let max_overlap = std::cmp::min(first_len, (second_len as f32 * 1.25) as usize);

    for overlap in 1..=max_overlap {
        let score = align_score(first, second, overlap, percent_time_overlap);
        if score > best_score {
            best_score = score;
            best_overlap = overlap;
        }
    }

    (best_overlap, best_score)
}

/// Weave together two transcript segments.
///
/// Intelligently combines overlapping transcription segments based on
/// Levenshtein distance-based alignment.
pub fn weave_transcript_segs(
    first_seg: &str,
    second_seg: &str,
    percent_time_overlap: f32,
    min_alignment_score: f32,
) -> String {
    let (overlap, a_score) = best_alignment(first_seg, second_seg, percent_time_overlap);

    if overlap == 0 || a_score < min_alignment_score {
        debug!(
            "No good alignment found, simple concatenation with score: {}",
            a_score
        );
        return format!("{} {}", first_seg, second_seg);
    }

    let mut best_score = 0.0;
    let mut best_trim = (0, 0);

    let first_chars: Vec<char> = first_seg.chars().collect();
    let second_chars: Vec<char> = second_seg.chars().collect();

    for idx in 0..=overlap {
        let left_start_idx = if idx >= overlap {
            0
        } else {
            first_chars.len().saturating_sub(overlap - idx)
        };

        let left = &first_seg[first_seg
            .char_indices()
            .nth(left_start_idx)
            .map_or(0, |(i, _)| i)..];

        for idx2 in 0..=overlap {
            let right_end_idx = std::cmp::min(overlap, second_chars.len());
            let right = &second_seg[..second_seg
                .char_indices()
                .nth(right_end_idx)
                .map_or_else(|| second_seg.len(), |(i, _)| i)];

            let adjusted_overlap = (overlap * 2).saturating_sub(idx + idx2);
            let score = trim_align_score(left, right, adjusted_overlap);

            if score > best_score {
                best_score = score;
                best_trim = (idx, idx2);
            }
        }
    }

    let first_trim_idx = if best_trim.0 >= overlap {
        first_seg.len()
    } else {
        let chars_to_keep = first_chars.len().saturating_sub(overlap - best_trim.0);
        first_seg
            .char_indices()
            .nth(chars_to_keep)
            .map_or_else(|| first_seg.len(), |(i, _)| i)
    };

    let second_trim_idx = second_seg
        .char_indices()
        .nth(best_trim.1)
        .map_or(0, |(i, _)| i);

    let result = format!(
        "{}{}",
        &first_seg[..first_trim_idx],
        &second_seg[second_trim_idx..]
    );
    debug!(
        "Weaved segments with overlap: {}, score: {}, result length: {}",
        overlap,
        best_score,
        result.len()
    );

    result
}

/// Detect if an overlap region contains silence.
///
/// Uses the audio energy to determine if the overlap is silent.
pub fn is_overlap_silence(overlap_audio: &[f32], mean_amplitude: f32) -> bool {
    if overlap_audio.is_empty() {
        return true;
    }

    // Calculate squared audio and smooth with convolution
    let squared_audio: Vec<f32> = overlap_audio.iter().map(|&s| s * s).collect();

    // Simple smoothing with a window of 800 (as in Python version)
    let window_size = std::cmp::min(800, squared_audio.len());
    let mut max_energy: f32 = 0.0;

    for i in 0..=squared_audio.len().saturating_sub(window_size) {
        let sum: f32 = squared_audio[i..i + window_size].iter().sum();
        let avg = sum / window_size as f32;
        max_energy = max_energy.max(avg);
    }

    let overlap_peak = max_energy.sqrt();
    info!(
        "overlap peak: {}, mean amplitude: {}",
        overlap_peak, mean_amplitude
    );

    overlap_peak < mean_amplitude / EXPECTED_SILENCE_RATIO
}
