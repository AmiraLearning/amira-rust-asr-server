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

#include "fst_cache.h"


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

    // Base decoding graph (general vocabulary)
    std::unique_ptr<k2::Fsa> base_graph_gpu_;

    // FST cache for per-user language models
    std::unique_ptr<FstCache> fst_cache_;

    // Configuration parameters
    float default_lm_weight_;      // Default LM interpolation weight
    bool enable_user_fsts_;        // Enable per-user FSTs
    int32_t expected_vocab_size_;  // Expected vocabulary size
    float search_beam_;            // Beam search parameter
    float output_beam_;            // Output beam parameter
    int32_t min_active_states_;    // Minimum active states
    int32_t max_active_states_;    // Maximum active states
};


K2DecoderModelInstance::K2DecoderModelInstance(
    TRITONBACKEND_ModelInstance* triton_model_instance)
    : BackendModelInstance(triton_model_instance),
      default_lm_weight_(0.5f),
      enable_user_fsts_(false),
      expected_vocab_size_(1030),
      search_beam_(20.0f),
      output_beam_(8.0f),
      min_active_states_(30),
      max_active_states_(10000)
{
    triton::common::TritonJson::Value params;
    if (!ModelConfig().Find("parameters", &params)) {
        throw BackendModelInstanceException(TRITONSERVER_ERROR_INVALID_ARG, "Parameters not found in model config");
    }

    // Load base decoding graph (required)
    triton::common::TritonJson::Value path_param;
    if (params.Find("DECODING_GRAPH_PATH", &path_param)) {
        std::string path_str;
        path_param.MemberAsString("string_value", &path_str);
        std::string full_path = JoinPath({Model()->RepositoryPath(), std::to_string(Model()->Version()), path_str});

        LOG_MESSAGE(TRITONSERVER_LOG_INFO, ("Loading base FST graph from " + full_path).c_str());

        std::ifstream is(full_path);
        if (!is.is_open()) {
            throw BackendModelInstanceException(TRITONSERVER_ERROR_NOT_FOUND, "Base FST graph not found at " + full_path);
        }

        auto base_graph_cpu = std::make_unique<k2::Fsa>(k2::Fsa::Read(is));
        base_graph_gpu_ = std::make_unique<k2::Fsa>(base_graph_cpu->To(torch::kCUDA, DeviceId()));

        LOG_MESSAGE(TRITONSERVER_LOG_INFO, "Base FST graph loaded successfully to GPU.");
    } else {
        throw BackendModelInstanceException(TRITONSERVER_ERROR_INVALID_ARG, "DECODING_GRAPH_PATH not found in model config");
    }

    // Load decoding parameters (optional - use defaults if not specified)
    triton::common::TritonJson::Value vocab_param;
    if (params.Find("VOCAB_SIZE", &vocab_param)) {
        std::string vocab_str;
        vocab_param.MemberAsString("string_value", &vocab_str);
        expected_vocab_size_ = std::stoi(vocab_str);
    }

    triton::common::TritonJson::Value search_beam_param;
    if (params.Find("SEARCH_BEAM", &search_beam_param)) {
        std::string beam_str;
        search_beam_param.MemberAsString("string_value", &beam_str);
        search_beam_ = std::stof(beam_str);
    }

    triton::common::TritonJson::Value output_beam_param;
    if (params.Find("OUTPUT_BEAM", &output_beam_param)) {
        std::string beam_str;
        output_beam_param.MemberAsString("string_value", &beam_str);
        output_beam_ = std::stof(beam_str);
    }

    triton::common::TritonJson::Value min_states_param;
    if (params.Find("MIN_ACTIVE_STATES", &min_states_param)) {
        std::string states_str;
        min_states_param.MemberAsString("string_value", &states_str);
        min_active_states_ = std::stoi(states_str);
    }

    triton::common::TritonJson::Value max_states_param;
    if (params.Find("MAX_ACTIVE_STATES", &max_states_param)) {
        std::string states_str;
        max_states_param.MemberAsString("string_value", &states_str);
        max_active_states_ = std::stoi(states_str);
    }

    LOG_MESSAGE(TRITONSERVER_LOG_INFO,
        ("Decoder params: vocab=" + std::to_string(expected_vocab_size_) +
         ", search_beam=" + std::to_string(search_beam_) +
         ", output_beam=" + std::to_string(output_beam_)).c_str());

    // Load FST cache configuration (optional - for per-user personalization)
    triton::common::TritonJson::Value user_fst_dir_param;
    if (params.Find("USER_FST_DIR", &user_fst_dir_param)) {
        std::string user_fst_dir;
        user_fst_dir_param.MemberAsString("string_value", &user_fst_dir);

        // Get max cached FSTs (optional, default 100)
        size_t max_cached_fsts = 100;
        triton::common::TritonJson::Value max_cache_param;
        if (params.Find("MAX_CACHED_FSTS", &max_cache_param)) {
            std::string max_cache_str;
            max_cache_param.MemberAsString("string_value", &max_cache_str);
            max_cached_fsts = std::stoul(max_cache_str);
        }

        // Get default LM weight (optional, default 0.5)
        triton::common::TritonJson::Value lm_weight_param;
        if (params.Find("DEFAULT_LM_WEIGHT", &lm_weight_param)) {
            std::string lm_weight_str;
            lm_weight_param.MemberAsString("string_value", &lm_weight_str);
            default_lm_weight_ = std::stof(lm_weight_str);
        }

        // Initialize FST cache
        FstCache::CacheConfig cache_config;
        cache_config.fst_directory = user_fst_dir;
        cache_config.max_cached_fsts = max_cached_fsts;
        cache_config.device_id = DeviceId();
        cache_config.enable_telemetry = true;

        fst_cache_ = std::make_unique<FstCache>(cache_config);
        enable_user_fsts_ = true;

        LOG_MESSAGE(TRITONSERVER_LOG_INFO,
            ("Per-user FST personalization enabled: dir=" + user_fst_dir +
             ", max_cache=" + std::to_string(max_cached_fsts) +
             ", default_lm_weight=" + std::to_string(default_lm_weight_)).c_str());
    } else {
        LOG_MESSAGE(TRITONSERVER_LOG_INFO,
            "Per-user FST personalization disabled (USER_FST_DIR not configured)");
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
    // === Step 1: Extract encoder_outputs (required) ===
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

    int32_t batch_size = input_shape[0];

    // === Step 2: Extract user_id (optional) ===
    std::string user_id = "";
    if (enable_user_fsts_) {
        TRITONBACKEND_Input* user_id_tensor = nullptr;
        err = TRITONBACKEND_RequestInput(request, "user_id", &user_id_tensor);

        if (err == nullptr && user_id_tensor != nullptr) {
            // Get user_id tensor buffer (should be BYTES/STRING type)
            const void* user_id_buffer;
            uint64_t user_id_buffer_size;
            TRITONSERVER_MemoryType user_id_memory_type;
            int64_t user_id_memory_type_id;

            err = TRITONBACKEND_InputBuffer(user_id_tensor, 0, &user_id_buffer,
                &user_id_buffer_size, &user_id_memory_type, &user_id_memory_type_id);

            if (err == nullptr && user_id_buffer_size >= 4) {
                // Extract string from buffer (first 4 bytes = length, rest = string data)
                const char* str_data = static_cast<const char*>(user_id_buffer);
                uint32_t str_len = *reinterpret_cast<const uint32_t*>(str_data);

                // Validate string length doesn't exceed buffer
                if (str_len > 0 && (4 + str_len) <= user_id_buffer_size) {
                    user_id = std::string(str_data + 4, str_len);

                    // Validate user_id doesn't contain path traversal
                    if (user_id.find("..") != std::string::npos ||
                        user_id.find("/") != std::string::npos ||
                        user_id.find("\\") != std::string::npos) {
                        LOG_MESSAGE(TRITONSERVER_LOG_WARN,
                            ("Invalid user_id contains path characters: " + user_id).c_str());
                        user_id = "";
                    } else {
                        LOG_MESSAGE(TRITONSERVER_LOG_DEBUG,
                            ("Processing request with user_id=" + user_id).c_str());
                    }
                } else {
                    LOG_MESSAGE(TRITONSERVER_LOG_WARN,
                        "Invalid user_id string format (length exceeds buffer), ignoring");
                }
            }
        }

        // Clear error if user_id not provided (it's optional)
        if (err != nullptr) {
            TRITONSERVER_ErrorDelete(err);
            err = nullptr;
        }
    }

    // === Step 3: Extract lm_weight (optional) ===
    float lm_weight = default_lm_weight_;
    TRITONBACKEND_Input* lm_weight_tensor = nullptr;
    err = TRITONBACKEND_RequestInput(request, "lm_weight", &lm_weight_tensor);

    if (err == nullptr && lm_weight_tensor != nullptr) {
        const void* lm_weight_buffer;
        uint64_t lm_weight_buffer_size;
        TRITONSERVER_MemoryType lm_weight_memory_type;
        int64_t lm_weight_memory_type_id;

        err = TRITONBACKEND_InputBuffer(lm_weight_tensor, 0, &lm_weight_buffer,
            &lm_weight_buffer_size, &lm_weight_memory_type, &lm_weight_memory_type_id);

        if (err == nullptr && lm_weight_buffer_size >= sizeof(float)) {
            lm_weight = *static_cast<const float*>(lm_weight_buffer);

            // Validate lm_weight is in valid range [0.0, 1.0]
            if (lm_weight < 0.0f || lm_weight > 1.0f) {
                LOG_MESSAGE(TRITONSERVER_LOG_WARN,
                    ("Invalid lm_weight=" + std::to_string(lm_weight) +
                     " (must be [0.0, 1.0]), clamping to valid range").c_str());
                lm_weight = std::clamp(lm_weight, 0.0f, 1.0f);
            }

            LOG_MESSAGE(TRITONSERVER_LOG_DEBUG,
                ("Using custom lm_weight=" + std::to_string(lm_weight)).c_str());
        }
    }

    // Clear error if lm_weight not provided (it's optional)
    if (err != nullptr) {
        TRITONSERVER_ErrorDelete(err);
        err = nullptr;
    }

    // === Step 4: Get decoding graph (base or composed with user FST) ===
    std::shared_ptr<k2::Fsa> user_fst_gpu = nullptr;
    const k2::Fsa* decoding_graph = base_graph_gpu_.get();

    if (enable_user_fsts_ && !user_id.empty()) {
        user_fst_gpu = fst_cache_->GetOrLoad(user_id);

        if (user_fst_gpu) {
            // TODO: Implement weighted composition here when k2 supports it
            // For now, just use the user FST directly (can be enhanced later)
            decoding_graph = user_fst_gpu.get();

            LOG_MESSAGE(TRITONSERVER_LOG_DEBUG,
                ("Using personalized FST for user_id=" + user_id).c_str());
        } else {
            LOG_MESSAGE(TRITONSERVER_LOG_DEBUG,
                ("User FST not found for user_id=" + user_id + ", using base graph").c_str());
        }
    }

    // === Step 5: Process logits ===
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(torch::kCUDA, DeviceId());

    // Validate input dimensions match expected vocab size
    int32_t vocab_size = input_shape[2];
    if (vocab_size != expected_vocab_size_) {
        err = TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG,
            ("Expected vocab size " + std::to_string(expected_vocab_size_) +
             ", got " + std::to_string(vocab_size)).c_str());
        RESPOND_AND_RETURN_IF_ERROR(response, err);
        return;
    }

    // Create tensor from raw logits
    // IMPORTANT: Clone tensor to ensure ownership of memory (Triton may free input_buffer)
    torch::Tensor raw_logits;
    try {
        // Create tensor on CPU first (from Triton-managed buffer)
        auto cpu_tensor = torch::from_blob(
            const_cast<void*>(input_buffer),
            {input_shape[0], input_shape[1], input_shape[2]},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
        );

        // Copy to GPU and clone to take ownership of data
        raw_logits = cpu_tensor.to(torch::kCUDA, DeviceId()).clone();

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
    std::vector<const k2::Fsa*> fsa_vec;
    for (int32_t b = 0; b < batch_size; ++b) {
        fsa_vec.push_back(decoding_graph);
    }
    k2::FsaVec decoding_graph_vec(fsa_vec);

    // Perform intersect dense pruned with error handling
    k2::FsaVec lattice;
    k2::FsaVec best_paths;
    try {
        lattice = k2::IntersectDensePruned(
            decoding_graph_vec, dense_fsa, search_beam_,
            output_beam_, min_active_states_, max_active_states_);
        
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