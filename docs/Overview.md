# Overview

AMIRA Rust ASR Server is a high-performance, real-time speech recognition server built in Rust with NVIDIA Triton integration. This overview orients you to the repo and points to deeper docs.

- What it is: Real-time and batch ASR with RNN-T models via Triton
- Key features: SIMD audio, lock-free pools, zero-copy paths, specialized executors, robust error handling, observability
- APIs: REST for batch, WebSocket for streaming

See also:
- [Architecture](./Architecture.md)
- [Operations](./Operations.md)
- [Performance Guide](./Performance.md)
- [Models](./Models.md)
- [Roadmap](./Roadmap.md)
