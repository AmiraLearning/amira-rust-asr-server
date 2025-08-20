# Architecture

## Components

- ASR core (`src/asr/`): pipeline, decoder, audio, SIMD, lock-free pools, zero-copy
- Triton integration (`src/triton/`): connection pool, reliable client, model IO
- Server (`src/server/`): REST handlers, WebSocket streaming
- Performance (`src/performance/`): CPU affinity, NUMA, specialized executors
- Reliability (`src/reliability/`): circuit breaker, metrics, tracing
- Platform (`src/platform/`): detection, capabilities, init

## Data Flow

```
Audio → Preprocessor → Encoder → Decoder/Joint → Text
   ↓         ↓            ↓            ↓
RingBuf  Features      Encoded      Token probs
```

## Error Handling

Typed errors via `thiserror` and `AppError` funnel; `IntoResponse` for HTTP; metrics hooks.

## Concurrency & Runtimes

Specialized executors: IO, Inference, Network; spawn per pool; avoid blocking on runtime.

## Memory & Performance

- Lock-free pools for buffers and workspaces
- Zero-copy where safe; CUDA path behind feature
- SIMD kernels (AVX2/AVX-512) with safe fallbacks

## Observability

- Metrics (`metrics-exporter-prometheus`) at `/metrics`
- Tracing configured via `tracing-subscriber`

## Config

Layered via Figment: env + yaml/toml + defaults. See `src/config.rs` and `config/*.yml`.
