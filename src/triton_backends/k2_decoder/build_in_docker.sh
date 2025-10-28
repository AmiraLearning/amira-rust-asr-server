#!/bin/bash
# Build k2_decoder Triton backend in Docker
#
# This script builds the k2_decoder backend in a containerized environment
# with all necessary dependencies (CUDA, k2, Triton).
#
# Usage:
#   ./build_in_docker.sh                # Build in Docker
#   ./build_in_docker.sh --interactive  # Start interactive shell for debugging

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="k2-decoder-builder"

echo "=================================================="
echo "Building k2_decoder Triton Backend in Docker"
echo "=================================================="

# Parse arguments
INTERACTIVE=0
if [[ "$1" == "--interactive" ]]; then
    INTERACTIVE=1
fi

# Build Docker image
echo "Building Docker image..."
docker build -f "${SCRIPT_DIR}/Dockerfile.build" -t "${IMAGE_NAME}" "${SCRIPT_DIR}"

if [[ $INTERACTIVE -eq 1 ]]; then
    echo "Starting interactive shell..."
    docker run --rm -it \
        -v "${SCRIPT_DIR}":/workspace \
        --gpus all \
        "${IMAGE_NAME}" bash
else
    # Run build
    echo "Running build..."
    docker run --rm \
        -v "${SCRIPT_DIR}":/workspace \
        --gpus all \
        "${IMAGE_NAME}"

    # Check if build succeeded
    if [[ -f "${SCRIPT_DIR}/build/libk2_decoder.so" ]]; then
        echo ""
        echo "✅ Build successful!"
        echo "Output: ${SCRIPT_DIR}/build/libk2_decoder.so"
        echo ""
        echo "Next steps:"
        echo "  1. Copy to Triton backends directory:"
        echo "     cp build/libk2_decoder.so /opt/tritonserver/backends/k2_decoder/"
        echo ""
        echo "  2. Configure Triton model:"
        echo "     cp config.pbtxt.example /models/k2_decoder/config.pbtxt"
        echo "     # Edit config.pbtxt to set USER_FST_DIR"
    else
        echo ""
        echo "❌ Build failed! Check output above for errors."
        exit 1
    fi
fi
