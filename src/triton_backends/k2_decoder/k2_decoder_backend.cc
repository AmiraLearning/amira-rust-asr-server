#include <memory>
#include <string>
#include <vector>
#include <fstream>

#include "k2/csrc/fsa.h"
#include "k2/csrc/intersect.h"
#include "k2/csrc/pytorch_utils.h"
#include "k2/csrc/properties.h"
#include "k2/csrc/util.h"
#include "k2/csrc/fsa_algo.h"

#include <torch/script.h>
#include <torch/torch.h>

#include "triton/backend/backend_common.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"


namespace triton { namespace backend { namespace k2_decoder {

#define RESPOND_AND_RETURN_IF_ERROR(RESPONSE, PP_ERROR)                 \
  do {                                                                  \
    if ((PP_ERROR) != nullptr) {                                        \
      TRITONBACKEND_ResponseSend(                                       \
          (RESPONSE), TRITONSERVER_RESPONSE_COMPLETE_FINAL, (PP_ERROR)); \
      return;                                                           \
    }                                                                   \
  } while (false)

class K2DecoderModelInstance : public BackendModelInstance {
public:
    static TRITONSERVER_Error* Create(
        K2DecoderModelInstance** instance,
        TRITONBACKEND_ModelInstance* triton_model_instance)
    {
        try {
            *instance = new K2DecoderModelInstance(triton_model_instance);
        } catch (const BackendModelInstanceException& ex) {
            return TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INTERNAL, ex.what());
        }
        return nullptr;
    }

    ~K2DecoderModelInstance() = default;

    TRITONSERVER_Error* Execute(
        TRITONBACKEND_Request** requests, const uint32_t request_count);

private:
    K2DecoderModelInstance(TRITONBACKEND_ModelInstance* triton_model_instance);

    void ProcessRequest(
        TRITONBACKEND_Request* request,
        TRITONBACKEND_Response* response);

    std::unique_ptr<k2::Fsa> decoding_graph_gpu_;
};


K2DecoderModelInstance::K2DecoderModelInstance(
    TRITONBACKEND_ModelInstance* triton_model_instance)
    : BackendModelInstance(triton_model_instance)
{
    // Load the FST graph path from the model configuration
    triton::common::TritonJson::Value params;
    if (ModelConfig().Find("parameters", &params)) {
        triton::common::TritonJson::Value path_param;
        if (params.Find("DECODING_GRAPH_PATH", &path_param)) {
            std::string path_str;
            path_param.MemberAsString("string_value", &path_str);
            std::string full_path = JoinPath({Model()->RepositoryPath(), std::to_string(Model()->Version()), path_str});
            
            LOG_MESSAGE(TRITONSERVER_LOG_INFO, ("Loading FST graph from " + full_path).c_str());

            std::ifstream is(full_path);
            if (!is.is_open()) {
                throw BackendModelInstanceException(TRITONSERVER_ERROR_NOT_FOUND, "FST graph not found at " + full_path);
            }
            
            auto decoding_graph_cpu = std::make_unique<k2::Fsa>(k2::Fsa::Read(is));
            decoding_graph_gpu_ = std::make_unique<k2::Fsa>(decoding_graph_cpu->To(torch::kCUDA, DeviceId()));
            
            LOG_MESSAGE(TRITONSERVER_LOG_INFO, "FST graph loaded successfully to GPU.");
        }
    } else {
        throw BackendModelInstanceException(TRITONSERVER_ERROR_INVALID_ARG, "DECODING_GRAPH_PATH not found in model config");
    }
}

TRITONSERVER_Error* K2DecoderModelInstance::Execute(
    TRITONBACKEND_Request** requests, const uint32_t request_count)
{
    for (uint32_t i = 0; i < request_count; ++i) {
        TRITONBACKEND_Request* request = requests[i];

        TRITONBACKEND_Response* response;
        auto* err = TRITONBACKEND_ResponseNew(&response, request);
        if (err != nullptr) {
            LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "Failed to create response");
            TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL);
            continue;
        }

        ProcessRequest(request, response);

        TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL);
    }
    return nullptr;
}

void K2DecoderModelInstance::ProcessRequest(
    TRITONBACKEND_Request* request,
    TRITONBACKEND_Response* response)
{
    TRITONBACKEND_Input* input_tensor;
    TRITONSERVER_Error* err = TRITONBACKEND_RequestInput(request, "encoder_outputs", &input_tensor);
    RESPOND_AND_RETURN_IF_ERROR(response, err);

    const int64_t* input_shape;
    uint32_t input_dims_count;
    TRITONSERVER_DataType input_datatype;
    err = TRITONBACKEND_InputProperties(input_tensor, nullptr, &input_datatype, &input_shape, &input_dims_count, nullptr, nullptr);
    RESPOND_AND_RETURN_IF_ERROR(response, err);

    if (input_dims_count != 3) {
        err = TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, "Input tensor 'encoder_outputs' must be 3-dimensional");
        RESPOND_AND_RETURN_IF_ERROR(response, err);
        return;
    }

    const void* input_buffer;
    uint64_t buffer_byte_size;
    TRITONSERVER_MemoryType input_memory_type;
    int64_t input_memory_type_id;

    err = TRITONBACKEND_InputBuffer(input_tensor, 0, &input_buffer, &buffer_byte_size, &input_memory_type, &input_memory_type_id);
    RESPOND_AND_RETURN_IF_ERROR(response, err);

    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(torch::kCUDA, DeviceId());
        
    // Validate input dimensions match expected vocab size
    int32_t vocab_size = input_shape[2];
    if (vocab_size != 1030) { // Expected vocab size from config
        err = TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, 
            ("Expected vocab size 1030, got " + std::to_string(vocab_size)).c_str());
        RESPOND_AND_RETURN_IF_ERROR(response, err);
        return;
    }
    
    // Create tensor from raw logits
    torch::Tensor raw_logits;
    try {
        raw_logits = torch::from_blob(const_cast<void*>(input_buffer), {input_shape[0], input_shape[1], input_shape[2]}, options);
        
        // Convert logits to log probabilities (required by k2::DenseFsaVec)
        // Using numerically stable log_softmax
        raw_logits = torch::log_softmax(raw_logits, /*dim=*/2);
    } catch (const std::exception& e) {
        err = TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, 
            ("Failed to create tensor from input: " + std::string(e.what())).c_str());
        RESPOND_AND_RETURN_IF_ERROR(response, err);
        return;
    }
    
    // Create DenseFsaVec from log probabilities
    k2::DenseFsaVec dense_fsa(raw_logits);
    
    // Replicate FSA for each batch item
    int32_t batch_size = input_shape[0];
    std::vector<const k2::Fsa*> fsa_vec;
    for (int32_t b = 0; b < batch_size; ++b) {
        fsa_vec.push_back(decoding_graph_gpu_.get());
    }
    k2::FsaVec decoding_graph_vec(fsa_vec);

    // Beam search parameters (TODO: make configurable via model parameters)
    float search_beam = 20.0;
    float output_beam = 8.0;
    int32_t min_active_states = 30;
    int32_t max_active_states = 10000;

    // Perform intersect dense pruned with error handling
    k2::FsaVec lattice;
    k2::FsaVec best_paths;
    try {
        lattice = k2::IntersectDensePruned(
            decoding_graph_vec, dense_fsa, search_beam,
            output_beam, min_active_states, max_active_states);
        
        // Get best paths for all batch items
        best_paths = k2::ShortestPath(lattice);
    } catch (const std::exception& e) {
        err = TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, 
            ("k2 decoding failed: " + std::string(e.what())).c_str());
        RESPOND_AND_RETURN_IF_ERROR(response, err);
        return;
    }
    
    // Move to CPU for token extraction
    auto best_paths_cpu = best_paths.To(torch::kCPU);

    // Extract tokens for all batch items
    std::vector<std::vector<int32_t>> batch_tokens(batch_size);
    int32_t max_seq_len = 0;
    
    for (int32_t b = 0; b < batch_size; ++b) {
        auto& best_path = best_paths_cpu[b];
        std::vector<int32_t> tokens;
        
        // Extract non-epsilon tokens from arcs
        for (int32_t i = 0; i < best_path.NumArcs(); ++i) {
            const auto& arc = best_path.GetArc(i);
            if (arc.label != 0 && arc.label != -1) { // Skip epsilon (0) and end-of-sentence (-1)
                tokens.push_back(arc.label);
            }
        }
        
        batch_tokens[b] = std::move(tokens);
        max_seq_len = std::max(max_seq_len, static_cast<int32_t>(batch_tokens[b].size()));
    }
    
    // Create output tensor with proper batch dimension
    TRITONBACKEND_Output* output_tensor;
    int64_t output_shape[] = {batch_size, max_seq_len};
    
    err = TRITONBACKEND_ResponseOutput(response, "tokens", TRITONSERVER_TYPE_INT32, output_shape, 2, &output_tensor);
    RESPOND_AND_RETURN_IF_ERROR(response, err);

    void* output_buffer;
    TRITONSERVER_MemoryType output_memory_type = TRITONSERVER_MEMORY_CPU;
    int64_t output_memory_type_id = 0;
    
    err = TRITONBACKEND_OutputBuffer(
        output_tensor, &output_buffer, batch_size * max_seq_len * sizeof(int32_t),
        &output_memory_type, &output_memory_type_id);
    RESPOND_AND_RETURN_IF_ERROR(response, err);

    // Fill output buffer with proper padding
    int32_t* output_tokens = static_cast<int32_t*>(output_buffer);
    std::fill(output_tokens, output_tokens + batch_size * max_seq_len, 0); // Pad with zeros
    
    for (int32_t b = 0; b < batch_size; ++b) {
        const auto& tokens = batch_tokens[b];
        std::copy(tokens.begin(), tokens.end(), output_tokens + b * max_seq_len);
    }

    TRITONBACKEND_ResponseSend(response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr);
}

extern "C" {

TRITONSERVER_Error* TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance) {
    K2DecoderModelInstance* model_instance;
    TRITONSERVER_Error* error = K2DecoderModelInstance::Create(&model_instance, instance);
    if (error != nullptr) {
        return error;
    }
    return TRITONBACKEND_ModelInstanceSetState(instance, reinterpret_cast<void*>(model_instance));
}

TRITONSERVER_Error* TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
    const uint32_t request_count)
{
    K2DecoderModelInstance* model_instance;
    TRITONSERVER_Error* err = TRITONBACKEND_ModelInstanceState(instance, reinterpret_cast<void**>(&model_instance));
    if (err != nullptr) {
        return err;
    }
    return model_instance->Execute(requests, request_count);
}

TRITONSERVER_Error* TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance) {
    void* vstate;
    TRITONSERVER_Error* err = TRITONBACKEND_ModelInstanceState(instance, &vstate);
    if (err != nullptr) {
        return err;
    }
    if (vstate != nullptr) {
        auto model_instance = reinterpret_cast<K2DecoderModelInstance*>(vstate);
        delete model_instance;
    }
    return nullptr;
}

} // extern "C"
} // namespace k2_decoder
} // namespace backend
} // namespace triton 