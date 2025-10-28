//! Low-level SIMD intrinsics and approximation functions
//!
//! This module provides fast approximations for mathematical functions
//! optimized for SIMD execution.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Fast exponential approximation for AVX2.
///
/// Uses polynomial approximation: exp(x) ≈ 1 + x + x²/2 + x³/6 for small x
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn exp_approx_avx2(x: __m256) -> __m256 {
    // Fast exp approximation using polynomial
    // exp(x) ≈ 1 + x + x²/2 + x³/6 for small x
    let one = _mm256_set1_ps(1.0);
    let half = _mm256_set1_ps(0.5);
    let sixth = _mm256_set1_ps(1.0 / 6.0);

    let x2 = _mm256_mul_ps(x, x);
    let x3 = _mm256_mul_ps(x2, x);

    let term1 = x;
    let term2 = _mm256_mul_ps(x2, half);
    let term3 = _mm256_mul_ps(x3, sixth);

    let result = _mm256_add_ps(one, term1);
    let result = _mm256_add_ps(result, term2);
    _mm256_add_ps(result, term3)
}

/// Fast exponential approximation for AVX-512.
///
/// Uses polynomial approximation: exp(x) ≈ 1 + x + x²/2 + x³/6 for small x
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn exp_approx_avx512(x: __m512) -> __m512 {
    // Fast exp approximation using polynomial
    // exp(x) ≈ 1 + x + x²/2 + x³/6 for small x
    let one = _mm512_set1_ps(1.0);
    let half = _mm512_set1_ps(0.5);
    let sixth = _mm512_set1_ps(1.0 / 6.0);

    let x2 = _mm512_mul_ps(x, x);
    let x3 = _mm512_mul_ps(x2, x);

    let term1 = x;
    let term2 = _mm512_mul_ps(x2, half);
    let term3 = _mm512_mul_ps(x3, sixth);

    let result = _mm512_add_ps(one, term1);
    let result = _mm512_add_ps(result, term2);
    _mm512_add_ps(result, term3)
}
