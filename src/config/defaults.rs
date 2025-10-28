//! Default value functions for configuration fields.
//!
//! These functions provide default values for serde deserialization,
//! allowing configuration fields to be optional in config files.

pub(crate) fn default_max_concurrent_streams() -> usize {
    10
}

pub(crate) fn default_max_concurrent_batches() -> usize {
    50
}

pub(crate) fn default_inference_queue_size() -> usize {
    100
}

pub(crate) fn default_audio_buffer_capacity() -> usize {
    1024 * 1024
} // 1MB

pub(crate) fn default_max_batch_audio_length() -> f32 {
    30.0
}

pub(crate) fn default_stream_timeout_secs() -> u64 {
    30
}

pub(crate) fn default_keepalive_check_period_ms() -> u64 {
    100
}

pub(crate) fn default_preprocessor_model_name() -> String {
    "preprocessor".to_string()
}

pub(crate) fn default_encoder_model_name() -> String {
    "encoder".to_string()
}

pub(crate) fn default_decoder_joint_model_name() -> String {
    "decoder_joint".to_string()
}

pub(crate) fn default_max_symbols_per_step() -> usize {
    30
}

pub(crate) fn default_max_total_tokens() -> usize {
    200
}

pub(crate) fn default_enable_platform_optimizations() -> bool {
    true
}

pub(crate) fn default_disable_numa_in_cloud() -> bool {
    true
}

pub(crate) fn default_disable_cpu_affinity() -> bool {
    false
}

pub(crate) fn default_force_io_uring() -> bool {
    false
}

pub(crate) fn default_inference_backend() -> String {
    "grpc".to_string()
}

pub(crate) fn default_cuda_device_id() -> i32 {
    0
}
