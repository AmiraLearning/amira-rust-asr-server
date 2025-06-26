# Multi-stage build for optimized production image
FROM rust:1.75-bullseye as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    protobuf-compiler \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy dependency files
COPY Cargo.toml Cargo.lock build.rs ./
COPY proto/ ./proto/

# Create dummy src to cache dependencies
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release && rm -rf src

# Copy actual source code
COPY src/ ./src/

# Build the application
RUN cargo build --release

# Runtime stage
FROM debian:bullseye-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl1.1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -s /bin/bash asr

# Copy binary from builder stage
COPY --from=builder /app/target/release/amira-asr-server /usr/local/bin/

# Set ownership and permissions
RUN chown asr:asr /usr/local/bin/amira-asr-server
USER asr

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8057/health || exit 1

# Expose port
EXPOSE 8057

# Default environment
ENV RUST_LOG=info
ENV SERVER_HOST=0.0.0.0
ENV SERVER_PORT=8057

# Run the binary
CMD ["amira-asr-server"]