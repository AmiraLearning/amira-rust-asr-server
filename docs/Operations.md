# Operations

## Configuration

- Files: `config.toml`, `config.yaml`
- Env: `AMIRA_*` variables (see README)
- Load order: env > yaml > toml > defaults

## Running

- Dev: `cargo run`
- Release: `cargo run --release`
- With config: `AMIRA_CONFIG_FILE=my-config.toml cargo run --release`

## Docker

- Build: `docker build -t amira-asr-server .`
- Compose: `docker-compose up -d`

## Monitoring

- Metrics: `GET /metrics` (Prometheus)
- Health: `GET /health`, `GET /health/detailed`
- Tracing: configure Jaeger endpoints via env

## Deployment Notes

- Non-root containers, minimal images
- Configure thread counts and affinity per host
- For cloud VMs, affinity may be disabled automatically
