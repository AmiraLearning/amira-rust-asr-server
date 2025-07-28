#!/bin/bash

# This script automates the download and installation of libtorch and k2
# for building the Triton C++ backend.
#
# Prerequisites:
# - A C++ compiler (g++)
# - git
# - cmake
# - wget or curl
# - unzip

set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
# You can change these versions if needed.
# For PyTorch versions, see: https://pytorch.org/get-started/previous-versions/
PYTORCH_VERSION="2.3.1"
TRITON_SERVER_VERSION="v2.59.0" # Corresponds to container 25.06

OS_TYPE=$(uname)
if [ "$OS_TYPE" == "Linux" ]; then
  OS="linux"
  CUDA_VERSION="12.1" # IMPORTANT: Change this to your NVIDIA GPU's CUDA version
elif [ "$OS_TYPE" == "Darwin" ]; then
  OS="macos"
  CUDA_VERSION="cpu" # No NVIDIA CUDA on macOS, will use CPU
else
  print_error "Unsupported OS: $OS_TYPE"
fi


# Installation directory
INSTALL_DIR="$(pwd)/third_party"
LIBTORCH_DIR="${INSTALL_DIR}/libtorch"
K2_DIR="${INSTALL_DIR}/k2"

# --- Helper Functions ---
Color_Off='\033[0m'
BGreen='\033[1;32m'
BYellow='\033[1;33m'
BRed='\033[1;31m'

function print_success() {
  echo -e "${BGreen}[SUCCESS]${Color_Off} $1"
}

function print_warning() {
  echo -e "${BYellow}[WARNING]${Color_Off} $1"
}

function print_error() {
  echo -e "${BRed}[ERROR]${Color_Off} $1"
  exit 1
}

function check_command() {
  if ! command -v "$1" &> /dev/null; then
    return 1
  fi
  return 0
}

function download_and_extract() {
    local url="$1"
    local filename="$2"
    echo "Downloading $filename from $url"
    if check_command wget; then
        wget --progress=bar:force -O "$filename" "$url"
    elif check_command curl; then
        curl -L --progress-bar -o "$filename" "$url"
    else
        print_error "This should not happen. Could not find wget or curl."
    fi
    if [ $? -ne 0 ]; then
        print_error "Failed to download $filename. Please check the URL."
    fi
    echo "Unzipping $filename..."
    unzip -q "$filename" -d "$INSTALL_DIR"
    rm "$filename"
    echo "Successfully extracted $filename to $INSTALL_DIR"
}

# --- Main Logic ---

# 1. Check for required system dependencies
echo "--- Step 1: Checking for required tools ---"
check_command git || print_error "'git' command not found. Please install it."
check_command cmake || print_error "'cmake' command not found. Please install it."
check_command unzip || print_error "'unzip' command not found. Please install it."
check_command g++ || print_error "'g++' C++ compiler not found. Please install build-essential or equivalent."

if ! check_command wget && ! check_command curl; then
    print_error "'wget' or 'curl' command not found. Please install one of them."
fi

print_success "All required tools are installed."
echo ""

# 2.5 Install PyTorch for Python (required by k2's build system)
echo "--- Step 2.5: Installing PyTorch for Python ---"
if python3 -c "import torch" &> /dev/null; then
    print_warning "PyTorch Python package already installed. Skipping."
else
    echo "Installing PyTorch for Python..."
    pip3 install torch torchvision torchaudio
    print_success "PyTorch for Python installed."
fi
echo ""

# 2. Create installation directory
echo "--- Step 2: Setting up installation directory ---"
mkdir -p "$INSTALL_DIR"
echo "Dependencies will be installed in: ${INSTALL_DIR}"
echo ""

# --- Step 3: Download and Unpack libtorch ---
print_info "Step 3: Downloading and Unpacking libtorch..."
if [ "$OS" == "linux" ]; then
    LIBTORCH_URL="https://download.pytorch.org/libtorch/cu${CUDA_VERSION//.}/libtorch-cxx11-abi-shared-with-deps-${PYTORCH_VERSION}%2Bcu${CUDA_VERSION//.}.zip"
elif [ "$OS" == "macos" ]; then
    LIBTORCH_URL="https://download.pytorch.org/libtorch/cpu/libtorch-macos-${PYTORCH_VERSION}.zip"
fi
download_and_extract "$LIBTORCH_URL" "libtorch.zip"

# --- Step 4: Download and Unpack Triton Server C-API ---
print_info "Step 4: Downloading and Unpacking Triton Server..."
if [ "$OS" == "linux" ]; then
    TRITON_URL="https://github.com/triton-inference-server/server/releases/download/${TRITON_SERVER_VERSION}/${TRITON_SERVER_VERSION}_ubuntu2204.clients.tar.gz"
    download_and_extract "$TRITON_URL" "tritonserver.tar.gz"
    # The Triton release asset name might be different, adjust if needed
    # For example, it might be named v2.59.0_ubuntu2204.clients.tar.gz
    # The script assumes a consistent naming, but check the release page if it fails.
    # We will create a 'tritonserver' directory to standardize the path
    mkdir -p tritonserver
    # Move the extracted contents (lib, include) into the tritonserver directory
    mv lib tritonserver/
    mv include tritonserver/

elif [ "$OS" == "macos" ]; then
    # Note: There is no official Triton C-API binary release for macOS.
    # This section is a placeholder. For a real macOS build, you would
    # need to compile Triton from source, which is a complex process.
    # For now, we will create dummy directories so the Rust build script can find them,
    # but linking will fail. This setup is for code completion and dev on mac.
    print_warning "No official Triton C-API binaries for macOS."
    print_warning "Creating dummy directories for local development."
    mkdir -p tritonserver/lib
    mkdir -p tritonserver/include
fi


# --- Step 5: Clone and Build k2 from Source ---
print_info "Step 5: Cloning and Building k2..."
K2_DIR="k2"
if [ -d "$K2_DIR" ]; then
    print_warning "K2 source directory already exists. Skipping git clone."
else
    echo "Cloning K2 repository..."
    git clone https://github.com/k2-fsa/k2.git "$K2_DIR"
fi

echo "Configuring K2 build..."
mkdir -p "$K2_DIR/src/build"
cd "$K2_DIR/src/build"

# Configure k2, pointing it to our downloaded libtorch.
# We also set CMAKE_INSTALL_PREFIX to keep the installation self-contained.
cmake -DCMAKE_INSTALL_PREFIX=./install \
      -DCMAKE_BUILD_TYPE=Release \
      -DK2_WITH_CUDA=ON \
      -DCMAKE_TOOLCHAIN_FILE=../libtorch/share/cmake/Caffe2/Caffe2Targets.cmake \
      ..

echo "Building K2 (this may take a while)..."
make -j$(nproc) install

cd ../../.. # Return to the project root

print_success "K2 installed successfully in ${K2_DIR}/install"
echo ""


# --- Step 6: Final Instructions ---
print_info "--- Step 6: Setup Complete! ---"
echo ""
print_success "Dependencies installed in: $INSTALL_DIR"

export CMAKE_PREFIX_PATH=$INSTALL_DIR/libtorch:$INSTALL_DIR/k2/install:$INSTALL_DIR/tritonserver
echo "To build your project, use this CMAKE_PREFIX_PATH:"
echo "export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH"
echo ""