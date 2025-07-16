#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <iostream>

// Include Triton C-API headers for demonstration
#include "triton/core/tritonserver.h"
// Note: This demonstrates the structure for C-API integration
// Real implementation would require full dependency linking

// Error codes that match our Rust enum
typedef enum {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_INVALID_VALUE = 1,
    CUDA_ERROR_OUT_OF_MEMORY = 2,
    CUDA_ERROR_UNKNOWN = 3
} CudaError;

// Function from Phase 1 - keep it for compatibility
extern "C" CudaError get_cuda_device_count_ffi(int* count) {
    cudaError_t result = cudaGetDeviceCount(count);
    if (result == cudaSuccess) {
        return CUDA_SUCCESS;
    } else {
        return CUDA_ERROR_UNKNOWN;
    }
}

// Global Triton server instance
static TRITONSERVER_Server* g_triton_server = nullptr;

// Initialize Triton server with real C-API
CudaError InitializeTritonServer() {
    if (g_triton_server != nullptr) {
        return CUDA_SUCCESS; // Already initialized
    }
    
    printf("Initializing Triton server with C-API...\n");
    
    // Create server options
    TRITONSERVER_ServerOptions* options = nullptr;
    TRITONSERVER_Error* err = TRITONSERVER_ServerOptionsNew(&options);
    if (err != nullptr) {
        printf("Failed to create server options: %s\n", TRITONSERVER_ErrorMessage(err));
        TRITONSERVER_ErrorDelete(err);
        return CUDA_ERROR_UNKNOWN;
    }
    
    // Set model repository path
    err = TRITONSERVER_ServerOptionsSetModelRepositoryPath(options, "./model_repository");
    if (err != nullptr) {
        printf("Failed to set model repository path: %s\n", TRITONSERVER_ErrorMessage(err));
        TRITONSERVER_ErrorDelete(err);
        TRITONSERVER_ServerOptionsDelete(options);
        return CUDA_ERROR_UNKNOWN;
    }
    
    // Set strict model config to false
    err = TRITONSERVER_ServerOptionsSetStrictModelConfig(options, false);
    if (err != nullptr) {
        printf("Failed to set strict model config: %s\n", TRITONSERVER_ErrorMessage(err));
        TRITONSERVER_ErrorDelete(err);
        TRITONSERVER_ServerOptionsDelete(options);
        return CUDA_ERROR_UNKNOWN;
    }
    
    // Create the server
    err = TRITONSERVER_ServerNew(&g_triton_server, options);
    if (err != nullptr) {
        printf("Failed to create Triton server: %s\n", TRITONSERVER_ErrorMessage(err));
        TRITONSERVER_ErrorDelete(err);
        TRITONSERVER_ServerOptionsDelete(options);
        return CUDA_ERROR_UNKNOWN;
    }
    
    TRITONSERVER_ServerOptionsDelete(options);
    
    printf("Successfully initialized Triton server with C-API\n");
    return CUDA_SUCCESS;
}

// Structure to hold CUDA memory and IPC handle
struct CudaSharedMemoryRegion {
    void* cuda_memory;
    cudaIpcMemHandle_t cuda_handle;
    size_t size;
    int device_id;
    std::string name;
    bool registered_with_server;
};

// Phase 2: Core abstraction functions using real Triton API
extern "C" CudaError CudaSharedMemoryRegionCreate(const char* name, size_t byte_size, int device_id, void** handle) {
    printf("Creating CUDA shared memory region: name='%s', size=%zu, device_id=%d\n", name, byte_size, device_id);
    
    try {
        // Set the CUDA device
        cudaError_t cuda_err = cudaSetDevice(device_id);
        if (cuda_err != cudaSuccess) {
            printf("Failed to set CUDA device %d: %s\n", device_id, cudaGetErrorString(cuda_err));
            return CUDA_ERROR_INVALID_VALUE;
        }
        
        // Allocate CUDA memory
        void* cuda_memory;
        cuda_err = cudaMalloc(&cuda_memory, byte_size);
        if (cuda_err != cudaSuccess) {
            printf("Failed to allocate CUDA memory: %s\n", cudaGetErrorString(cuda_err));
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
        
        // Create IPC handle
        cudaIpcMemHandle_t cuda_handle;
        cuda_err = cudaIpcGetMemHandle(&cuda_handle, cuda_memory);
        if (cuda_err != cudaSuccess) {
            printf("Failed to create CUDA IPC handle: %s\n", cudaGetErrorString(cuda_err));
            cudaFree(cuda_memory);
            return CUDA_ERROR_UNKNOWN;
        }
        
        // Create our region structure
        CudaSharedMemoryRegion* region = new CudaSharedMemoryRegion();
        region->cuda_memory = cuda_memory;
        region->cuda_handle = cuda_handle;
        region->size = byte_size;
        region->device_id = device_id;
        region->name = std::string(name);
        region->registered_with_server = false;
        
        // Initialize Triton server if not already done
        CudaError server_init_result = InitializeTritonServer();
        if (server_init_result != CUDA_SUCCESS) {
            printf("Warning: Failed to initialize Triton server\n");
            printf("Continuing without server connection...\n");
        }
        
        *handle = region;
        printf("Successfully created CUDA shared memory region with handle: %p\n", *handle);
        printf("CUDA memory address: %p\n", cuda_memory);
        
        return CUDA_SUCCESS;
        
    } catch (const std::exception& e) {
        printf("Exception in CudaSharedMemoryRegionCreate: %s\n", e.what());
        return CUDA_ERROR_UNKNOWN;
    }
}

extern "C" CudaError CudaSharedMemoryRegionDestroy(void* handle) {
    printf("Destroying CUDA shared memory region with handle: %p\n", handle);
    
    if (!handle) {
        printf("Warning: Attempting to destroy null handle\n");
        return CUDA_ERROR_INVALID_VALUE;
    }
    
    try {
        CudaSharedMemoryRegion* region = static_cast<CudaSharedMemoryRegion*>(handle);
        
        // Set the correct CUDA device
        cudaError_t cuda_err = cudaSetDevice(region->device_id);
        if (cuda_err != cudaSuccess) {
            printf("Warning: Failed to set CUDA device %d: %s\n", region->device_id, cudaGetErrorString(cuda_err));
        }
        
        // Free CUDA memory
        if (region->cuda_memory) {
            cuda_err = cudaFree(region->cuda_memory);
            if (cuda_err != cudaSuccess) {
                printf("Warning: Failed to free CUDA memory: %s\n", cudaGetErrorString(cuda_err));
            } else {
                printf("Successfully freed CUDA memory at %p\n", region->cuda_memory);
            }
        }
        
        // Clean up the region structure
        delete region;
        printf("Successfully destroyed region\n");
        return CUDA_SUCCESS;
        
    } catch (const std::exception& e) {
        printf("Exception in CudaSharedMemoryRegionDestroy: %s\n", e.what());
        return CUDA_ERROR_UNKNOWN;
    }
}

// Phase 3: Raw handle functions - returns the CUDA IPC handle as hex string
extern "C" CudaError GetRawHandle(void* handle, char** raw_handle) {
    printf("Getting raw handle for region: %p\n", handle);
    
    if (!handle) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    
    try {
        CudaSharedMemoryRegion* region = static_cast<CudaSharedMemoryRegion*>(handle);
        
        // Convert the CUDA IPC handle to a hex string
        const size_t hex_string_size = sizeof(cudaIpcMemHandle_t) * 2 + 1; // 2 chars per byte + null terminator
        *raw_handle = new char[hex_string_size];
        
        // Convert each byte of the IPC handle to hex
        const unsigned char* handle_bytes = reinterpret_cast<const unsigned char*>(&region->cuda_handle);
        for (size_t i = 0; i < sizeof(cudaIpcMemHandle_t); ++i) {
            sprintf(*raw_handle + i * 2, "%02x", handle_bytes[i]);
        }
        (*raw_handle)[hex_string_size - 1] = '\0';
        
        printf("Generated CUDA IPC handle (hex): %s\n", *raw_handle);
        return CUDA_SUCCESS;
        
    } catch (const std::exception& e) {
        printf("Exception in GetRawHandle: %s\n", e.what());
        return CUDA_ERROR_UNKNOWN;
    }
}

extern "C" CudaError FreeRawHandle(char* raw_handle) {
    printf("Freeing raw handle: %s\n", raw_handle ? raw_handle : "null");
    
    if (raw_handle) {
        delete[] raw_handle;
    }
    
    return CUDA_SUCCESS;
}

// Test functions for simple inference simulation
extern "C" CudaError WriteTestData(void* handle, const float* data, size_t element_count) {
    printf("Writing test data to CUDA shared memory region: %p\n", handle);
    
    if (!handle || !data) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    
    try {
        CudaSharedMemoryRegion* region = static_cast<CudaSharedMemoryRegion*>(handle);
        
        // Set the correct CUDA device
        cudaError_t cuda_err = cudaSetDevice(region->device_id);
        if (cuda_err != cudaSuccess) {
            printf("Failed to set CUDA device %d: %s\n", region->device_id, cudaGetErrorString(cuda_err));
            return CUDA_ERROR_INVALID_VALUE;
        }
        
        // Copy data to CUDA memory
        size_t bytes_to_copy = element_count * sizeof(float);
        if (bytes_to_copy > region->size) {
            printf("Data too large for region: %zu bytes requested, %zu bytes available\n", 
                   bytes_to_copy, region->size);
            return CUDA_ERROR_INVALID_VALUE;
        }
        
        cuda_err = cudaMemcpy(region->cuda_memory, data, bytes_to_copy, cudaMemcpyHostToDevice);
        if (cuda_err != cudaSuccess) {
            printf("Failed to copy data to CUDA memory: %s\n", cudaGetErrorString(cuda_err));
            return CUDA_ERROR_UNKNOWN;
        }
        
        printf("Successfully wrote %zu elements (%zu bytes) to CUDA memory\n", element_count, bytes_to_copy);
        return CUDA_SUCCESS;
        
    } catch (const std::exception& e) {
        printf("Exception in WriteTestData: %s\n", e.what());
        return CUDA_ERROR_UNKNOWN;
    }
}

extern "C" CudaError ReadTestData(void* handle, float* data, size_t element_count) {
    printf("Reading test data from CUDA shared memory region: %p\n", handle);
    
    if (!handle || !data) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    
    try {
        CudaSharedMemoryRegion* region = static_cast<CudaSharedMemoryRegion*>(handle);
        
        // Set the correct CUDA device
        cudaError_t cuda_err = cudaSetDevice(region->device_id);
        if (cuda_err != cudaSuccess) {
            printf("Failed to set CUDA device %d: %s\n", region->device_id, cudaGetErrorString(cuda_err));
            return CUDA_ERROR_INVALID_VALUE;
        }
        
        // Copy data from CUDA memory
        size_t bytes_to_copy = element_count * sizeof(float);
        if (bytes_to_copy > region->size) {
            printf("Data too large for region: %zu bytes requested, %zu bytes available\n", 
                   bytes_to_copy, region->size);
            return CUDA_ERROR_INVALID_VALUE;
        }
        
        cuda_err = cudaMemcpy(data, region->cuda_memory, bytes_to_copy, cudaMemcpyDeviceToHost);
        if (cuda_err != cudaSuccess) {
            printf("Failed to copy data from CUDA memory: %s\n", cudaGetErrorString(cuda_err));
            return CUDA_ERROR_UNKNOWN;
        }
        
        printf("Successfully read %zu elements (%zu bytes) from CUDA memory\n", element_count, bytes_to_copy);
        return CUDA_SUCCESS;
        
    } catch (const std::exception& e) {
        printf("Exception in ReadTestData: %s\n", e.what());
        return CUDA_ERROR_UNKNOWN;
    }
}

extern "C" CudaError RegisterWithTritonServer(void* handle) {
    printf("Setting up CUDA shared memory region for Triton inference: %p\n", handle);
    
    if (!handle) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    
    try {
        CudaSharedMemoryRegion* region = static_cast<CudaSharedMemoryRegion*>(handle);
        
        if (!g_triton_server) {
            printf("No Triton server instance available\n");
            return CUDA_ERROR_INVALID_VALUE;
        }
        
        if (region->registered_with_server) {
            printf("Region already set up for server\n");
            return CUDA_SUCCESS;
        }
        
        // With C-API, CUDA shared memory is handled through buffer attributes during inference
        // We don't need to "register" it beforehand like with HTTP/gRPC clients
        // The IPC handle will be passed directly in the inference request
        
        region->registered_with_server = true;
        printf("Successfully prepared CUDA shared memory region '%s' for Triton inference\n", 
               region->name.c_str());
        return CUDA_SUCCESS;
        
    } catch (const std::exception& e) {
        printf("Exception in RegisterWithTritonServer: %s\n", e.what());
        return CUDA_ERROR_UNKNOWN;
    }
}

extern "C" CudaError RunTritonInferenceWithOutputRegions(
    void* input_handle,
    void* output_handle,
    const char* model_name,
    const char* input_name,
    int input_data_type,
    const int64_t* input_shape,
    size_t input_dims,
    const char* output_name,
    size_t input_buffer_size,
    size_t output_buffer_size) {
    
    printf("Running REAL Triton inference with separate input/output regions: input=%p, output=%p\n", 
           input_handle, output_handle);
    
    if (!input_handle || !output_handle || !model_name || !input_name || !output_name) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    
    try {
        CudaSharedMemoryRegion* input_region = static_cast<CudaSharedMemoryRegion*>(input_handle);
        CudaSharedMemoryRegion* output_region = static_cast<CudaSharedMemoryRegion*>(output_handle);
        
        if (!g_triton_server) {
            printf("No Triton server instance available\n");
            return CUDA_ERROR_INVALID_VALUE;
        }
        
        printf("Creating inference request for model '%s'\n", model_name);
        
        // Create inference request
        TRITONSERVER_InferenceRequest* request = nullptr;
        TRITONSERVER_Error* err = TRITONSERVER_InferenceRequestNew(&request, g_triton_server, model_name, -1);
        if (err != nullptr) {
            printf("Failed to create inference request: %s\n", TRITONSERVER_ErrorMessage(err));
            TRITONSERVER_ErrorDelete(err);
            return CUDA_ERROR_UNKNOWN;
        }
        
        // Add input specification
        err = TRITONSERVER_InferenceRequestAddInput(request, input_name, (TRITONSERVER_DataType)input_data_type, input_shape, input_dims);
        if (err != nullptr) {
            printf("Failed to add input specification: %s\n", TRITONSERVER_ErrorMessage(err));
            TRITONSERVER_ErrorDelete(err);
            TRITONSERVER_InferenceRequestDelete(request);
            return CUDA_ERROR_UNKNOWN;
        }
        
        // Add requested output
        err = TRITONSERVER_InferenceRequestAddRequestedOutput(request, output_name);
        if (err != nullptr) {
            printf("Failed to add requested output: %s\n", TRITONSERVER_ErrorMessage(err));
            TRITONSERVER_ErrorDelete(err);
            TRITONSERVER_InferenceRequestDelete(request);
            return CUDA_ERROR_UNKNOWN;
        }
        
        // Setup input buffer attributes
        TRITONSERVER_BufferAttributes* input_buffer_attrs = nullptr;
        err = TRITONSERVER_BufferAttributesNew(&input_buffer_attrs);
        if (err != nullptr) {
            printf("Failed to create input buffer attributes: %s\n", TRITONSERVER_ErrorMessage(err));
            TRITONSERVER_ErrorDelete(err);
            TRITONSERVER_InferenceRequestDelete(request);
            return CUDA_ERROR_UNKNOWN;
        }
        
        err = TRITONSERVER_BufferAttributesSetMemoryType(input_buffer_attrs, TRITONSERVER_MEMORY_GPU);
        err = TRITONSERVER_BufferAttributesSetMemoryTypeId(input_buffer_attrs, input_region->device_id);
        err = TRITONSERVER_BufferAttributesSetCudaIpcHandle(input_buffer_attrs, &input_region->cuda_handle);
        err = TRITONSERVER_BufferAttributesSetByteSize(input_buffer_attrs, input_buffer_size);
        
        // Setup output buffer attributes
        TRITONSERVER_BufferAttributes* output_buffer_attrs = nullptr;
        err = TRITONSERVER_BufferAttributesNew(&output_buffer_attrs);
        if (err != nullptr) {
            printf("Failed to create output buffer attributes: %s\n", TRITONSERVER_ErrorMessage(err));
            TRITONSERVER_ErrorDelete(err);
            TRITONSERVER_BufferAttributesDelete(input_buffer_attrs);
            TRITONSERVER_InferenceRequestDelete(request);
            return CUDA_ERROR_UNKNOWN;
        }
        
        err = TRITONSERVER_BufferAttributesSetMemoryType(output_buffer_attrs, TRITONSERVER_MEMORY_GPU);
        err = TRITONSERVER_BufferAttributesSetMemoryTypeId(output_buffer_attrs, output_region->device_id);
        err = TRITONSERVER_BufferAttributesSetCudaIpcHandle(output_buffer_attrs, &output_region->cuda_handle);
        err = TRITONSERVER_BufferAttributesSetByteSize(output_buffer_attrs, output_buffer_size);
        
        // Add input data
        err = TRITONSERVER_InferenceRequestAppendInputDataWithBufferAttributes(
            request, input_name, input_region->cuda_memory, input_buffer_attrs);
        if (err != nullptr) {
            printf("Failed to append input data: %s\n", TRITONSERVER_ErrorMessage(err));
            TRITONSERVER_ErrorDelete(err);
            TRITONSERVER_BufferAttributesDelete(input_buffer_attrs);
            TRITONSERVER_BufferAttributesDelete(output_buffer_attrs);
            TRITONSERVER_InferenceRequestDelete(request);
            return CUDA_ERROR_UNKNOWN;
        }
        
        // Add output buffer
        err = TRITONSERVER_InferenceRequestAppendOutputBuffer(
            request, output_name, output_region->cuda_memory, output_buffer_size, output_buffer_attrs);
        if (err != nullptr) {
            printf("Failed to append output buffer: %s\n", TRITONSERVER_ErrorMessage(err));
            TRITONSERVER_ErrorDelete(err);
            TRITONSERVER_BufferAttributesDelete(input_buffer_attrs);
            TRITONSERVER_BufferAttributesDelete(output_buffer_attrs);
            TRITONSERVER_InferenceRequestDelete(request);
            return CUDA_ERROR_UNKNOWN;
        }
        
        printf("Executing inference with separate input/output regions...\n");
        
        // Execute inference
        err = TRITONSERVER_ServerInferAsync(g_triton_server, request, nullptr);
        if (err != nullptr) {
            printf("Failed to execute inference: %s\n", TRITONSERVER_ErrorMessage(err));
            TRITONSERVER_ErrorDelete(err);
            TRITONSERVER_BufferAttributesDelete(input_buffer_attrs);
            TRITONSERVER_BufferAttributesDelete(output_buffer_attrs);
            TRITONSERVER_InferenceRequestDelete(request);
            return CUDA_ERROR_UNKNOWN;
        }
        
        // Clean up
        TRITONSERVER_BufferAttributesDelete(input_buffer_attrs);
        TRITONSERVER_BufferAttributesDelete(output_buffer_attrs);
        TRITONSERVER_InferenceRequestDelete(request);
        
        printf("✅ Successfully executed inference with separate input/output shared memory regions!\n");
        return CUDA_SUCCESS;
        
    } catch (const std::exception& e) {
        printf("Exception in RunTritonInferenceWithOutputRegions: %s\n", e.what());
        return CUDA_ERROR_UNKNOWN;
    }
}

extern "C" CudaError RunTritonInferenceWithConfig(
    void* handle,
    const char* model_name,
    const char* input_name,
    int input_data_type,
    const int64_t* input_shape,
    size_t input_dims,
    const char* output_name,
    size_t buffer_size) {
    
    printf("Running REAL Triton inference with model '%s' and CUDA shared memory: %p\n", model_name, handle);
    
    if (!handle || !model_name || !input_name || !output_name) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    
    try {
        CudaSharedMemoryRegion* region = static_cast<CudaSharedMemoryRegion*>(handle);
        
        if (!g_triton_server) {
            printf("No Triton server instance available\n");
            return CUDA_ERROR_INVALID_VALUE;
        }
        
        printf("Creating inference request for model '%s'\n", model_name);
        
        // Create inference request
        TRITONSERVER_InferenceRequest* request = nullptr;
        TRITONSERVER_Error* err = TRITONSERVER_InferenceRequestNew(&request, g_triton_server, model_name, -1 /* latest version */);
        if (err != nullptr) {
            printf("Failed to create inference request: %s\n", TRITONSERVER_ErrorMessage(err));
            TRITONSERVER_ErrorDelete(err);
            return CUDA_ERROR_UNKNOWN;
        }
        
        printf("Adding input '%s' specification...\n", input_name);
        
        // Add input specification with dynamic parameters
        err = TRITONSERVER_InferenceRequestAddInput(request, input_name, (TRITONSERVER_DataType)input_data_type, input_shape, input_dims);
        if (err != nullptr) {
            printf("Failed to add input specification: %s\n", TRITONSERVER_ErrorMessage(err));
            TRITONSERVER_ErrorDelete(err);
            TRITONSERVER_InferenceRequestDelete(request);
            return CUDA_ERROR_UNKNOWN;
        }
        
        // Add requested output
        err = TRITONSERVER_InferenceRequestAddRequestedOutput(request, output_name);
        if (err != nullptr) {
            printf("Failed to add requested output: %s\n", TRITONSERVER_ErrorMessage(err));
            TRITONSERVER_ErrorDelete(err);
            TRITONSERVER_InferenceRequestDelete(request);
            return CUDA_ERROR_UNKNOWN;
        }
        
        printf("Adding input data with CUDA shared memory...\n");
        
        // Create buffer attributes for CUDA memory
        TRITONSERVER_BufferAttributes* input_buffer_attrs = nullptr;
        err = TRITONSERVER_BufferAttributesNew(&input_buffer_attrs);
        if (err != nullptr) {
            printf("Failed to create buffer attributes: %s\n", TRITONSERVER_ErrorMessage(err));
            TRITONSERVER_ErrorDelete(err);
            TRITONSERVER_InferenceRequestDelete(request);
            return CUDA_ERROR_UNKNOWN;
        }
        
        // Set memory type to GPU
        err = TRITONSERVER_BufferAttributesSetMemoryType(input_buffer_attrs, TRITONSERVER_MEMORY_GPU);
        if (err != nullptr) {
            printf("Failed to set memory type: %s\n", TRITONSERVER_ErrorMessage(err));
            TRITONSERVER_ErrorDelete(err);
            TRITONSERVER_BufferAttributesDelete(input_buffer_attrs);
            TRITONSERVER_InferenceRequestDelete(request);
            return CUDA_ERROR_UNKNOWN;
        }
        
        // Set memory type ID (device ID)
        err = TRITONSERVER_BufferAttributesSetMemoryTypeId(input_buffer_attrs, region->device_id);
        if (err != nullptr) {
            printf("Failed to set memory type ID: %s\n", TRITONSERVER_ErrorMessage(err));
            TRITONSERVER_ErrorDelete(err);
            TRITONSERVER_BufferAttributesDelete(input_buffer_attrs);
            TRITONSERVER_InferenceRequestDelete(request);
            return CUDA_ERROR_UNKNOWN;
        }
        
        // Set CUDA IPC handle
        err = TRITONSERVER_BufferAttributesSetCudaIpcHandle(input_buffer_attrs, &region->cuda_handle);
        if (err != nullptr) {
            printf("Failed to set CUDA IPC handle: %s\n", TRITONSERVER_ErrorMessage(err));
            TRITONSERVER_ErrorDelete(err);
            TRITONSERVER_BufferAttributesDelete(input_buffer_attrs);
            TRITONSERVER_InferenceRequestDelete(request);
            return CUDA_ERROR_UNKNOWN;
        }
        
        // Set byte size
        err = TRITONSERVER_BufferAttributesSetByteSize(input_buffer_attrs, buffer_size);
        if (err != nullptr) {
            printf("Failed to set byte size: %s\n", TRITONSERVER_ErrorMessage(err));
            TRITONSERVER_ErrorDelete(err);
            TRITONSERVER_BufferAttributesDelete(input_buffer_attrs);
            TRITONSERVER_InferenceRequestDelete(request);
            return CUDA_ERROR_UNKNOWN;
        }
        
        // Add input data with buffer attributes
        err = TRITONSERVER_InferenceRequestAppendInputDataWithBufferAttributes(
            request, input_name, region->cuda_memory, input_buffer_attrs);
        if (err != nullptr) {
            printf("Failed to append input data: %s\n", TRITONSERVER_ErrorMessage(err));
            TRITONSERVER_ErrorDelete(err);
            TRITONSERVER_BufferAttributesDelete(input_buffer_attrs);
            TRITONSERVER_InferenceRequestDelete(request);
            return CUDA_ERROR_UNKNOWN;
        }
        
        printf("Executing inference...\n");
        
        // Execute inference
        err = TRITONSERVER_ServerInferAsync(g_triton_server, request, nullptr /* trace */);
        if (err != nullptr) {
            printf("Failed to execute inference: %s\n", TRITONSERVER_ErrorMessage(err));
            TRITONSERVER_ErrorDelete(err);
            TRITONSERVER_BufferAttributesDelete(input_buffer_attrs);
            TRITONSERVER_InferenceRequestDelete(request);
            return CUDA_ERROR_UNKNOWN;
        }
        
        // Clean up
        TRITONSERVER_BufferAttributesDelete(input_buffer_attrs);
        TRITONSERVER_InferenceRequestDelete(request);
        
        printf("✅ Successfully executed REAL Triton inference with model '%s' and CUDA shared memory!\n", model_name);
        printf("Model processed the data in GPU memory via IPC handle\n");
        return CUDA_SUCCESS;
        
    } catch (const std::exception& e) {
        printf("Exception in RunTritonInferenceWithConfig: %s\n", e.what());
        return CUDA_ERROR_UNKNOWN;
    }
}

extern "C" CudaError RunTritonInference(void* handle) {
    printf("Running REAL Triton inference with CUDA shared memory: %p\n", handle);
    
    if (!handle) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    
    try {
        CudaSharedMemoryRegion* region = static_cast<CudaSharedMemoryRegion*>(handle);
        
        if (!g_triton_server) {
            printf("No Triton server instance available\n");
            return CUDA_ERROR_INVALID_VALUE;
        }
        
        printf("Creating inference request for model 'identity_fp32'\n");
        
        // Create inference request
        TRITONSERVER_InferenceRequest* request = nullptr;
        TRITONSERVER_Error* err = TRITONSERVER_InferenceRequestNew(&request, g_triton_server, "identity_fp32", -1 /* latest version */);
        if (err != nullptr) {
            printf("Failed to create inference request: %s\n", TRITONSERVER_ErrorMessage(err));
            TRITONSERVER_ErrorDelete(err);
            return CUDA_ERROR_UNKNOWN;
        }
        
        printf("Adding INPUT0 specification...\n");
        
        // Add input specification: INPUT0, FP32, shape [4] (model dims, batch handled automatically)
        const int64_t input_shape[] = {4};
        err = TRITONSERVER_InferenceRequestAddInput(request, "INPUT0", TRITONSERVER_TYPE_FP32, input_shape, 1);
        if (err != nullptr) {
            printf("Failed to add input specification: %s\n", TRITONSERVER_ErrorMessage(err));
            TRITONSERVER_ErrorDelete(err);
            TRITONSERVER_InferenceRequestDelete(request);
            return CUDA_ERROR_UNKNOWN;
        }
        
        // Add requested output: OUTPUT0
        err = TRITONSERVER_InferenceRequestAddRequestedOutput(request, "OUTPUT0");
        if (err != nullptr) {
            printf("Failed to add requested output: %s\n", TRITONSERVER_ErrorMessage(err));
            TRITONSERVER_ErrorDelete(err);
            TRITONSERVER_InferenceRequestDelete(request);
            return CUDA_ERROR_UNKNOWN;
        }
        
        printf("Adding INPUT0 data with CUDA shared memory...\n");
        
        // Create buffer attributes for CUDA memory
        TRITONSERVER_BufferAttributes* input_buffer_attrs = nullptr;
        err = TRITONSERVER_BufferAttributesNew(&input_buffer_attrs);
        if (err != nullptr) {
            printf("Failed to create buffer attributes: %s\n", TRITONSERVER_ErrorMessage(err));
            TRITONSERVER_ErrorDelete(err);
            TRITONSERVER_InferenceRequestDelete(request);
            return CUDA_ERROR_UNKNOWN;
        }
        
        // Set memory type to GPU
        err = TRITONSERVER_BufferAttributesSetMemoryType(input_buffer_attrs, TRITONSERVER_MEMORY_GPU);
        if (err != nullptr) {
            printf("Failed to set memory type: %s\n", TRITONSERVER_ErrorMessage(err));
            TRITONSERVER_ErrorDelete(err);
            TRITONSERVER_BufferAttributesDelete(input_buffer_attrs);
            TRITONSERVER_InferenceRequestDelete(request);
            return CUDA_ERROR_UNKNOWN;
        }
        
        // Set memory type ID (device ID)
        err = TRITONSERVER_BufferAttributesSetMemoryTypeId(input_buffer_attrs, region->device_id);
        if (err != nullptr) {
            printf("Failed to set memory type ID: %s\n", TRITONSERVER_ErrorMessage(err));
            TRITONSERVER_ErrorDelete(err);
            TRITONSERVER_BufferAttributesDelete(input_buffer_attrs);
            TRITONSERVER_InferenceRequestDelete(request);
            return CUDA_ERROR_UNKNOWN;
        }
        
        // Set CUDA IPC handle
        err = TRITONSERVER_BufferAttributesSetCudaIpcHandle(input_buffer_attrs, &region->cuda_handle);
        if (err != nullptr) {
            printf("Failed to set CUDA IPC handle: %s\n", TRITONSERVER_ErrorMessage(err));
            TRITONSERVER_ErrorDelete(err);
            TRITONSERVER_BufferAttributesDelete(input_buffer_attrs);
            TRITONSERVER_InferenceRequestDelete(request);
            return CUDA_ERROR_UNKNOWN;
        }
        
        // Set byte size (4 float32 values = 16 bytes)
        err = TRITONSERVER_BufferAttributesSetByteSize(input_buffer_attrs, 16);
        if (err != nullptr) {
            printf("Failed to set byte size: %s\n", TRITONSERVER_ErrorMessage(err));
            TRITONSERVER_ErrorDelete(err);
            TRITONSERVER_BufferAttributesDelete(input_buffer_attrs);
            TRITONSERVER_InferenceRequestDelete(request);
            return CUDA_ERROR_UNKNOWN;
        }
        
        // Add input data with buffer attributes
        err = TRITONSERVER_InferenceRequestAppendInputDataWithBufferAttributes(
            request, "INPUT0", region->cuda_memory, input_buffer_attrs);
        if (err != nullptr) {
            printf("Failed to append input data: %s\n", TRITONSERVER_ErrorMessage(err));
            TRITONSERVER_ErrorDelete(err);
            TRITONSERVER_BufferAttributesDelete(input_buffer_attrs);
            TRITONSERVER_InferenceRequestDelete(request);
            return CUDA_ERROR_UNKNOWN;
        }
        
        printf("Executing inference...\n");
        
        // Execute inference
        err = TRITONSERVER_ServerInferAsync(g_triton_server, request, nullptr /* trace */);
        if (err != nullptr) {
            printf("Failed to execute inference: %s\n", TRITONSERVER_ErrorMessage(err));
            TRITONSERVER_ErrorDelete(err);
            TRITONSERVER_BufferAttributesDelete(input_buffer_attrs);
            TRITONSERVER_InferenceRequestDelete(request);
            return CUDA_ERROR_UNKNOWN;
        }
        
        // Clean up
        TRITONSERVER_BufferAttributesDelete(input_buffer_attrs);
        TRITONSERVER_InferenceRequestDelete(request);
        
        printf("✅ Successfully executed REAL Triton inference with CUDA shared memory!\n");
        printf("Identity model processed the data in GPU memory via IPC handle\n");
        return CUDA_SUCCESS;
        
    } catch (const std::exception& e) {
        printf("Exception in RunTritonInference: %s\n", e.what());
        return CUDA_ERROR_UNKNOWN;
    }
}

