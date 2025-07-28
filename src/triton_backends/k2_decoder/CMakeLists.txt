cmake_minimum_required(VERSION 3.18)
project(k2_decoder_backend)

# Set CMAKE_PREFIX_PATH to find libtorch and k2
set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/libtorch;${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/k2/install")

find_package(Torch REQUIRED)
find_package(k2 REQUIRED)

set(TRITON_COMMON_REPO_TAG "r25.06" CACHE STRING "Triton common repo tag")
set(TRITON_BACKEND_REPO_TAG "r25.06" CACHE STRING "Triton backend repo tag")
set(TRITON_CORE_REPO_TAG "r25.06" CACHE STRING "Triton core repo tag")

include(FetchContent)

FetchContent_Declare(
    triton_common
    GIT_REPOSITORY https://github.com/triton-inference-server/common.git
    GIT_TAG ${TRITON_COMMON_REPO_TAG}
)
FetchContent_MakeAvailable(triton_common)

FetchContent_Declare(
    triton_backend
    GIT_REPOSITORY https://github.com/triton-inference-server/backend.git
    GIT_TAG ${TRITON_BACKEND_REPO_TAG}
)
FetchContent_MakeAvailable(triton_backend)

add_library(k2_decoder SHARED k2_decoder_backend.cc)

target_link_libraries(k2_decoder
    PRIVATE
    ${TORCH_LIBRARIES}
    k2::k2_core
    triton-backend
)

target_include_directories(k2_decoder
    PRIVATE
    ${k2_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${triton_backend_SOURCE_DIR}/include
    ${triton_common_SOURCE_DIR}/include
)

set_target_properties(k2_decoder PROPERTIES
    CUDA_ARCHITECTURES "70;75;80;86;90"  # Added 90 for GH200 (H100/H200)
)

install(TARGETS k2_decoder DESTINATION backends/k2_decoder) 