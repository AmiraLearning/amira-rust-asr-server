# TODO: Remaining Optimizations

This file tracks remaining tasks from the original implementation plan. See `docs/legacy/TODO.md` for the full historical context.

## 🎯 High Priority

### 1. Voice Activity Detection (VAD) Pre-processor
**Priority**: High
**Estimated Effort**: 4-5 days
**Performance Impact**: 20-40% reduction in GPU utilization

**Current State:**
- `SILENCE_THRESHOLD` constant defined but unused (src/constants.rs:51)
- Mean amplitude calculation exists (src/asr/audio.rs:63-66)
- Only used for transcript weaving, not for inference gating

**Implementation:**
- [ ] Research VAD algorithm (WebRTC VAD, Silero VAD, or energy-based)
- [ ] Add VAD check before GPU inference calls
- [ ] Implement configurable sensitivity threshold
- [ ] Add VAD confidence scoring
- [ ] Create metrics for VAD skip rate
- [ ] Add tests for various audio scenarios (speech/silence/noise)

**Files to modify:**
- `src/asr/cuda_pipeline.rs`: Add VAD check before `infer()` calls
- `src/asr/incremental.rs`: Add VAD check in `process_chunk()`
- `src/constants.rs`: Make SILENCE_THRESHOLD configurable
- `src/config.rs`: Add VAD configuration options

---

## 🔧 Medium Priority

### 2. Thread-Local Memory Pools
**Priority**: Medium
**Estimated Effort**: 2-3 days
**Performance Impact**: ~0.01ms → ~0.001ms per pool access

**Current State:**
- Using global memory pools (src/asr/memory.rs)
- Pool access has atomic overhead

**Implementation:**
- [ ] Create thread-local audio buffer pool
- [ ] Create thread-local encoder buffer pool
- [ ] Add fallback to global pool when thread-local exhausted
- [ ] Benchmark thread-local vs global access patterns
- [ ] Add tests for thread-safety and cleanup

**Example:**
```rust
thread_local! {
    static LOCAL_AUDIO_BUFFER: RefCell<Vec<f32>> =
        RefCell::new(Vec::with_capacity(16000));
}
```

### 3. Sticky Triton Connections
**Priority**: Medium
**Estimated Effort**: 1-2 days
**Performance Impact**: ~0.1ms → ~0.01ms per connection access

**Current State:**
- Connection pool access on every inference
- No connection affinity per stream

**Implementation:**
- [ ] Add dedicated connection field to StreamProcessor
- [ ] Implement connection lease/borrow mechanism
- [ ] Add connection health check and auto-renewal
- [ ] Handle connection failures gracefully
- [ ] Add metrics for connection reuse rate

**Files to modify:**
- `src/server/stream.rs`: Add connection field to StreamProcessor
- `src/triton/pool.rs`: Add sticky lease mechanism

### 4. Lazy Audio Buffer Conversion
**Priority**: Medium
**Estimated Effort**: 1-2 days
**Performance Impact**: ~0.1ms → ~0.01ms for buffering

**Current State:**
- Immediate conversion from bytes to f32 samples
- Conversion happens even if audio might be buffered

**Implementation:**
- [ ] Store raw bytes in buffer
- [ ] Convert only when needed for inference
- [ ] Add caching layer for converted samples
- [ ] Invalidate cache on new data arrival
- [ ] Benchmark impact on real-world usage

---

## 📊 Low Priority / Nice-to-Have

### 5. In-Place Decoder State Updates
**Priority**: Low
**Estimated Effort**: 2-3 days
**Performance Impact**: ~1-2ms → ~0.5ms per decode step

**Current State:**
- Decoder state gets moved/cloned (needs verification)
- May cause unnecessary memory allocations

**Implementation:**
- [ ] Audit current decoder state handling in cuda_pipeline.rs
- [ ] Modify decoder to accept mutable state references
- [ ] Update inference calls to use in-place updates
- [ ] Benchmark memory allocation impact
- [ ] Ensure no regressions in accuracy

### 6. Enhanced Error Handling
**Priority**: Low
**Estimated Effort**: 3-4 days

**Remaining Work:**
- [ ] Add comprehensive SIMD feature detection with fallback tests
- [ ] Implement graceful degradation scenarios
- [ ] Add automatic recovery mechanisms for transient failures
- [ ] Create error recovery integration tests
- [ ] Document failure modes and recovery strategies

### 7. Additional Security Hardening
**Priority**: Low
**Estimated Effort**: 2-3 days

**Remaining Work:**
- [ ] Audit path validation for file operations
- [ ] Add comprehensive input sanitization for config values
- [ ] Implement API-level rate limiting (beyond WebSocket)
- [ ] Add security audit documentation
- [ ] Create threat model document

### 8. Testing Coverage Enhancements
**Priority**: Low
**Estimated Effort**: 4-5 days

**Current**: 311 tests passing (12 ignored)

**Remaining Work:**
- [ ] Add end-to-end integration tests for batch/streaming
- [ ] Create performance regression test suite
- [ ] Add memory leak detection tests
- [ ] Implement load testing scenarios
- [ ] Add cross-platform CI matrix
- [ ] Enable ignored GPU/Triton integration tests in CI

---

## 📈 Performance Impact Summary

| Optimization | Current | Target | Impact | Priority |
|--------------|---------|--------|--------|----------|
| **VAD Pre-processor** | 100% GPU usage | 60-80% GPU usage | **20-40% reduction** | **HIGH** |
| Thread-local pools | 0.01ms | 0.001ms | 0.009ms saved | Medium |
| Sticky connections | 0.1ms | 0.01ms | 0.09ms saved | Medium |
| Lazy audio conversion | 0.1ms | 0.01ms | 0.09ms saved | Medium |
| In-place state updates | 1-2ms | 0.5ms | 0.5-1.5ms saved | Low |

**Total potential improvement (excluding VAD)**: ~0.7-1.7ms per chunk
**Total with VAD**: 20-40% GPU utilization reduction + ~0.7-1.7ms latency

---

## 🚀 Quick Wins

If you have limited time, prioritize these in order:

1. **VAD Pre-processor** (4-5 days) - Biggest impact by far
2. **Sticky Triton Connections** (1-2 days) - Easy implementation, decent impact
3. **Thread-local Memory Pools** (2-3 days) - Good performance/effort ratio

---

## ⚡ Advanced/Expert Level (HFT Optimizations)

These are microsecond-level optimizations from `docs/legacy/HPC.md` inspired by high-frequency trading systems. Only pursue these after completing higher-priority items, or if you need extreme low-latency performance.

### 9. Error Aggregation Pattern
**Priority**: Expert
**Estimated Effort**: 3-4 days
**Performance Impact**: ~0.2ms → ~0.001ms (200x speedup)

**Current State:**
- Using `Result<>` with `?` operator throughout hotpath
- Multiple error checks introduce branching overhead

**Implementation:**
- [ ] Create `HotpathContext` with error flags
- [ ] Replace `Result<>` checks with flag setting in hotpath
- [ ] Aggregate error checking to single point
- [ ] Add comprehensive error flag tests
- [ ] Ensure no regressions in error handling

**Reference**: `docs/legacy/HPC.md` Section 1

### 10. Branch Elimination via Templates
**Priority**: Expert
**Estimated Effort**: 4-5 days
**Performance Impact**: ~0.1ms → ~0.02ms (5x speedup)

**Current State:**
- Runtime branching in decoder loop for blank token detection
- Conditional logic in hotpath

**Implementation:**
- [ ] Create `DecoderStrategy` trait for compile-time specialization
- [ ] Implement `BlankExpectedStrategy` and `TokenExpectedStrategy`
- [ ] Refactor decoder to use strategy pattern
- [ ] Benchmark impact on various audio types
- [ ] Ensure accuracy is maintained

**Reference**: `docs/legacy/HPC.md` Section 2

### 11. Cache Warming (Instruction Cache)
**Priority**: Expert
**Estimated Effort**: 2-3 days
**Performance Impact**: ~5ms cold → ~0.1ms warm (50x speedup on first call)

**Current State:**
- Cold cache on first inference
- No continuous warming mechanism

**Implementation:**
- [ ] Create `CacheWarmer` background task
- [ ] Run dummy inference continuously (without Triton call)
- [ ] Measure impact on first-request latency
- [ ] Add metrics for cache warming effectiveness
- [ ] Make warming configurable/disableable

**Reference**: `docs/legacy/HPC.md` Section 3

### 12. Cache-Line Aligned Data Structures
**Priority**: Expert
**Estimated Effort**: 3-4 days
**Performance Impact**: ~0.5ms → ~0.1ms (5x speedup)

**Current State:**
- No explicit cache-line alignment
- Data structures may span multiple cache lines
- **Current**: Runtime SIMD detection with `is_x86_feature_detected!` (10+ call sites)

**Implementation:**
- [ ] Add `#[repr(C, align(64))]` to hotpath structs
- [ ] Pack frequently-accessed data into 64-byte cache lines
- [ ] Use `std::ptr::copy_nonoverlapping` for efficient copies
- [ ] Benchmark memory access patterns
- [ ] Profile with perf/cachegrind

**Reference**: `docs/legacy/HPC.md` Section 4

### 13. Compile-Time SIMD Selection
**Priority**: Expert
**Estimated Effort**: 2-3 days
**Performance Impact**: ~0.05ms → ~0.01ms (5x speedup)

**Current State:**
- Runtime SIMD detection with `is_x86_feature_detected!`
- Found in 10+ locations in `src/asr/simd.rs`

**Implementation:**
- [ ] Use `#[cfg(target_feature)]` for compile-time dispatch
- [ ] Create separate build targets for AVX2/AVX512
- [ ] Remove runtime SIMD branching
- [ ] Update build documentation for target-specific builds
- [ ] Add CI jobs for different SIMD targets

**Reference**: `docs/legacy/HPC.md` Section 5

**Example:**
```bash
# Build for specific CPU features
cargo build --release --target-feature=+avx512f
```

---

## 🎯 HFT Performance Goals

If all HFT optimizations are implemented:

| Component | Current | Optimized | Speedup |
|-----------|---------|-----------|---------|
| Error handling | 0.2ms | 0.001ms | 200x |
| Branch elimination | 0.1ms | 0.02ms | 5x |
| Cache warming | 5ms (cold) | 0.1ms | 50x |
| Data layout | 0.5ms | 0.1ms | 5x |
| SIMD dispatch | 0.05ms | 0.01ms | 5x |

**Total hotpath improvement**: From ~25-30ms to ~15-20ms (30-40% faster)

**Note**: These optimizations are highly complex and should only be pursued if:
- You need predictable, consistent microsecond-level latency
- You have extensive systems programming experience
- You have proper benchmarking and profiling infrastructure
- You've exhausted all higher-priority optimizations

---

## Notes

- The original detailed TODO with code examples is preserved in `docs/legacy/TODO.md`
- HFT-level optimizations from CppCon 2017 talk are documented in `docs/legacy/HPC.md`
- Current roadmap focus is on production deployment (see `docs/Roadmap.md`)
- VAD was deprioritized in August 2024 but remains the highest-impact optimization
- Most infrastructure work (io_uring, cloud detection, CUDA integration) is complete
