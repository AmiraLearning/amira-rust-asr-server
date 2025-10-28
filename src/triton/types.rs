//! Triton tensor and model interface types.
//!
//! This module defines helper types for working with Triton tensors and models.

use crate::error::{AppError, AsrError, ModelError, Result};
use crate::triton::proto::{InferTensorContents, ModelInferResponse};
use bytes::Bytes;
use std::collections::HashMap;

/// Defines a tensor's shape (dimensions).
#[derive(Debug, Clone)]
pub struct TensorShape(pub Vec<i64>);

impl TensorShape {
    /// Create a new tensor shape with the given dimensions.
    pub fn new(dims: Vec<i64>) -> Self {
        Self(dims)
    }

    /// Create a scalar tensor shape.
    pub fn scalar() -> Self {
        Self(vec![])
    }

    /// Create a tensor shape for a vector (1D tensor).
    pub fn vector(length: i64) -> Self {
        Self(vec![length])
    }

    /// Create a tensor shape for a batch of vectors.
    pub fn batch_vector(batch_size: i64, length: i64) -> Self {
        Self(vec![batch_size, length])
    }

    /// Create a tensor shape for a 3D tensor (common for ASR models).
    pub fn batch_3d(batch_size: i64, channels: i64, time_len: i64) -> Self {
        Self(vec![batch_size, channels, time_len])
    }

    /// Get the dimensions of this tensor shape.
    pub fn dims(&self) -> &[i64] {
        &self.0
    }

    /// Calculate the total number of elements in this tensor.
    pub fn num_elements(&self) -> usize {
        self.0.iter().fold(1, |acc, &dim| acc * dim.max(1) as usize)
    }
}

/// Defines a tensor's data type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorDataType {
    /// 32-bit floating point
    Float32,

    /// 64-bit floating point
    Float64,

    /// 32-bit signed integer
    Int32,

    /// 64-bit signed integer
    Int64,

    /// 8-bit unsigned integer
    UInt8,

    /// Boolean
    Bool,
}

impl TensorDataType {
    /// Convert to Triton datatype string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Float32 => "FP32",
            Self::Float64 => "FP64",
            Self::Int32 => "INT32",
            Self::Int64 => "INT64",
            Self::UInt8 => "UINT8",
            Self::Bool => "BOOL",
        }
    }

    /// Get the size in bytes of a single element of this type.
    pub fn element_size(&self) -> usize {
        match self {
            Self::Float32 => 4,
            Self::Float64 => 8,
            Self::Int32 => 4,
            Self::Int64 => 8,
            Self::UInt8 => 1,
            Self::Bool => 1,
        }
    }
}

/// Defines a tensor's definition (name, datatype, shape).
#[derive(Debug, Clone)]
pub struct TensorDef {
    /// The tensor's name
    pub name: String,

    /// The tensor's data type
    pub dtype: TensorDataType,

    /// The tensor's shape
    pub shape: TensorShape,
}

impl TensorDef {
    /// Create a new tensor definition.
    pub fn new<S: Into<String>>(name: S, dtype: TensorDataType, shape: TensorShape) -> Self {
        Self {
            name: name.into(),
            dtype,
            shape,
        }
    }

    /// Calculate the expected size in bytes of this tensor.
    pub fn byte_size(&self) -> usize {
        self.shape.num_elements() * self.dtype.element_size()
    }
}

/// Enum for holding tensor data of different types.
#[derive(Debug, Clone)]
pub enum TensorData {
    /// 32-bit floating point data
    Float32(Vec<f32>),

    /// 64-bit floating point data
    Float64(Vec<f64>),

    /// 32-bit signed integer data
    Int32(Vec<i32>),

    /// 64-bit signed integer data
    Int64(Vec<i64>),

    /// 8-bit unsigned integer data
    UInt8(Vec<u8>),

    /// Boolean data
    Bool(Vec<bool>),
}

impl TensorData {
    /// Get the data type of this tensor data.
    pub fn dtype(&self) -> TensorDataType {
        match self {
            Self::Float32(_) => TensorDataType::Float32,
            Self::Float64(_) => TensorDataType::Float64,
            Self::Int32(_) => TensorDataType::Int32,
            Self::Int64(_) => TensorDataType::Int64,
            Self::UInt8(_) => TensorDataType::UInt8,
            Self::Bool(_) => TensorDataType::Bool,
        }
    }

    /// Get the number of elements in this tensor data.
    pub fn len(&self) -> usize {
        match self {
            Self::Float32(v) => v.len(),
            Self::Float64(v) => v.len(),
            Self::Int32(v) => v.len(),
            Self::Int64(v) => v.len(),
            Self::UInt8(v) => v.len(),
            Self::Bool(v) => v.len(),
        }
    }

    /// Check if this tensor data is empty.
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Float32(v) => v.is_empty(),
            Self::Float64(v) => v.is_empty(),
            Self::Int32(v) => v.is_empty(),
            Self::Int64(v) => v.is_empty(),
            Self::UInt8(v) => v.is_empty(),
            Self::Bool(v) => v.is_empty(),
        }
    }

    /// Convert to Triton InferTensorContents.
    pub fn to_contents(self) -> InferTensorContents {
        let mut contents = InferTensorContents::default();

        match self {
            Self::Float32(data) => contents.fp32_contents = data,
            Self::Float64(data) => contents.fp64_contents = data,
            Self::Int32(data) => contents.int_contents = data,
            Self::Int64(data) => contents.int64_contents = data,
            Self::UInt8(data) => {
                contents.uint_contents = data.into_iter().map(|x| x as u32).collect()
            }
            Self::Bool(data) => contents.bool_contents = data,
        }

        contents
    }
}

/// Represents a raw tensor from Triton's binary response format.
#[derive(Debug)]
pub struct RawTensor {
    /// The tensor's definition
    pub def: TensorDef,

    /// The tensor's raw binary data
    pub data: Bytes,
}

impl RawTensor {
    /// Parse the tensor's raw binary data as f32.
    pub fn as_f32(&self) -> Result<Vec<f32>> {
        if self.def.dtype != TensorDataType::Float32 {
            return Err(AppError::Asr(AsrError::ModelInference(
                ModelError::Inference(format!(
                    "Cannot parse tensor '{}' as f32, its type is {:?}",
                    self.def.name, self.def.dtype
                )),
            )));
        }

        let vec_len = self.data.len() / 4;
        let mut result = Vec::with_capacity(vec_len);

        for i in 0..vec_len {
            let offset = i * 4;
            let chunk: [u8; 4] = self.data[offset..offset + 4].try_into().map_err(|_| {
                AppError::Asr(AsrError::ModelInference(ModelError::Inference(format!(
                    "Failed to read f32 from tensor '{}'",
                    self.def.name
                ))))
            })?;
            result.push(f32::from_le_bytes(chunk));
        }

        Ok(result)
    }

    /// Parse the tensor's raw binary data as i32.
    pub fn as_i32(&self) -> Result<Vec<i32>> {
        if self.def.dtype != TensorDataType::Int32 {
            return Err(AppError::Asr(AsrError::ModelInference(
                ModelError::Inference(format!(
                    "Cannot parse tensor '{}' as i32, its type is {:?}",
                    self.def.name, self.def.dtype
                )),
            )));
        }

        let vec_len = self.data.len() / 4;
        let mut result = Vec::with_capacity(vec_len);

        for i in 0..vec_len {
            let offset = i * 4;
            let chunk: [u8; 4] = self.data[offset..offset + 4].try_into().map_err(|_| {
                AppError::Asr(AsrError::ModelInference(ModelError::Inference(format!(
                    "Failed to read i32 from tensor '{}'",
                    self.def.name
                ))))
            })?;
            result.push(i32::from_le_bytes(chunk));
        }

        Ok(result)
    }

    /// Parse the tensor's raw binary data as i64.
    pub fn as_i64(&self) -> Result<Vec<i64>> {
        if self.def.dtype != TensorDataType::Int64 {
            return Err(AppError::Asr(AsrError::ModelInference(
                ModelError::Inference(format!(
                    "Cannot parse tensor '{}' as i64, its type is {:?}",
                    self.def.name, self.def.dtype
                )),
            )));
        }

        let vec_len = self.data.len() / 8;
        let mut result = Vec::with_capacity(vec_len);

        for i in 0..vec_len {
            let offset = i * 8;
            let chunk: [u8; 8] = self.data[offset..offset + 8].try_into().map_err(|_| {
                AppError::Asr(AsrError::ModelInference(ModelError::Inference(format!(
                    "Failed to read i64 from tensor '{}'",
                    self.def.name
                ))))
            })?;
            result.push(i64::from_le_bytes(chunk));
        }

        Ok(result)
    }

    /// Get a single i64 value from the tensor.
    pub fn as_scalar_i64(&self) -> Result<i64> {
        let values = self.as_i64()?;
        values.first().copied().ok_or_else(|| {
            AppError::Asr(AsrError::ModelInference(ModelError::Inference(format!(
                "Tensor '{}' is empty",
                self.def.name
            ))))
        })
    }

    /// Get a single f32 value from the tensor.
    pub fn as_scalar_f32(&self) -> Result<f32> {
        let values = self.as_f32()?;
        values.first().copied().ok_or_else(|| {
            AppError::Asr(AsrError::ModelInference(ModelError::Inference(format!(
                "Tensor '{}' is empty",
                self.def.name
            ))))
        })
    }
}

/// Parse raw tensor data from Triton's binary response format.
///
/// # Arguments
/// * `response` - The Triton inference response
/// * `expected_tensors` - A map of tensor names to their expected definitions
///
/// # Returns
/// A map of tensor names to their parsed raw tensors
pub fn parse_raw_tensors(
    response: &ModelInferResponse,
    expected_tensors: &HashMap<String, TensorDef>,
) -> Result<HashMap<String, RawTensor>> {
    if response.raw_output_contents.is_empty() {
        return Err(AppError::Asr(AsrError::ModelInference(
            ModelError::Inference("Response does not contain raw tensor data".to_string()),
        )));
    }

    let mut result = HashMap::new();

    // First, build a map of tensor name to its index and size
    let mut tensor_indices = HashMap::new();

    for (i, output_tensor) in response.outputs.iter().enumerate() {
        let name = output_tensor.name.clone();

        // Always use the actual shape from the response for size calculation
        let shape: Vec<i64> = output_tensor.shape.clone();

        // Determine the data type
        let dtype = if let Some(tensor_def) = expected_tensors.get(&name) {
            // Use expected dtype if provided
            tensor_def.dtype
        } else {
            // Parse dtype from response
            match output_tensor.datatype.as_str() {
                "FP32" => TensorDataType::Float32,
                "FP64" => TensorDataType::Float64,
                "INT32" => TensorDataType::Int32,
                "INT64" => TensorDataType::Int64,
                "UINT8" => TensorDataType::UInt8,
                "BOOL" => TensorDataType::Bool,
                _ => {
                    return Err(AppError::Asr(AsrError::ModelInference(
                        ModelError::Inference(format!(
                            "Unsupported tensor datatype: {}",
                            output_tensor.datatype
                        )),
                    )));
                }
            }
        };

        // Create tensor definition with actual shape from response
        let tensor_def = TensorDef {
            name: name.clone(),
            dtype,
            shape: TensorShape(shape),
        };

        let byte_size = tensor_def.byte_size();
        tensor_indices.insert(name, (i, byte_size, tensor_def));
    }

    // Now, extract each tensor's data from the raw buffer
    for (name, (index, byte_size, tensor_def)) in tensor_indices {
        if index >= response.raw_output_contents.len() {
            return Err(AppError::Asr(AsrError::ModelInference(
                ModelError::Inference(format!(
                    "Tensor '{}' index out of bounds: {} >= {}",
                    name,
                    index,
                    response.raw_output_contents.len()
                )),
            )));
        }

        let data = Bytes::from(response.raw_output_contents[index].clone());

        if data.len() != byte_size {
            return Err(AppError::Asr(AsrError::ModelInference(
                ModelError::Inference(format!(
                    "Tensor '{}' size mismatch: expected {} bytes, got {}",
                    name,
                    byte_size,
                    data.len()
                )),
            )));
        }

        result.insert(
            name,
            RawTensor {
                def: tensor_def,
                data,
            },
        );
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    // TensorShape tests
    #[test]
    fn test_tensor_shape_new() {
        let shape = TensorShape::new(vec![2, 3, 4]);
        assert_eq!(shape.dims(), &[2, 3, 4]);
    }

    #[test]
    fn test_tensor_shape_scalar() {
        let shape = TensorShape::scalar();
        assert!(shape.dims().is_empty());
        assert_eq!(shape.num_elements(), 1);
    }

    #[test]
    fn test_tensor_shape_vector() {
        let shape = TensorShape::vector(10);
        assert_eq!(shape.dims(), &[10]);
        assert_eq!(shape.num_elements(), 10);
    }

    #[test]
    fn test_tensor_shape_batch_vector() {
        let shape = TensorShape::batch_vector(4, 256);
        assert_eq!(shape.dims(), &[4, 256]);
        assert_eq!(shape.num_elements(), 1024);
    }

    #[test]
    fn test_tensor_shape_batch_3d() {
        let shape = TensorShape::batch_3d(2, 3, 100);
        assert_eq!(shape.dims(), &[2, 3, 100]);
        assert_eq!(shape.num_elements(), 600);
    }

    #[test]
    fn test_tensor_shape_num_elements() {
        let shape = TensorShape::new(vec![2, 3, 4, 5]);
        assert_eq!(shape.num_elements(), 120);

        let empty_shape = TensorShape::new(vec![]);
        assert_eq!(empty_shape.num_elements(), 1); // Scalar has 1 element
    }

    // TensorDataType tests
    #[test]
    fn test_tensor_dtype_as_str() {
        assert_eq!(TensorDataType::Float32.as_str(), "FP32");
        assert_eq!(TensorDataType::Float64.as_str(), "FP64");
        assert_eq!(TensorDataType::Int32.as_str(), "INT32");
        assert_eq!(TensorDataType::Int64.as_str(), "INT64");
        assert_eq!(TensorDataType::UInt8.as_str(), "UINT8");
        assert_eq!(TensorDataType::Bool.as_str(), "BOOL");
    }

    #[test]
    fn test_tensor_dtype_element_size() {
        assert_eq!(TensorDataType::Float32.element_size(), 4);
        assert_eq!(TensorDataType::Float64.element_size(), 8);
        assert_eq!(TensorDataType::Int32.element_size(), 4);
        assert_eq!(TensorDataType::Int64.element_size(), 8);
        assert_eq!(TensorDataType::UInt8.element_size(), 1);
        assert_eq!(TensorDataType::Bool.element_size(), 1);
    }

    // TensorDef tests
    #[test]
    fn test_tensor_def_new() {
        let def = TensorDef::new(
            "test_tensor",
            TensorDataType::Float32,
            TensorShape::vector(100),
        );

        assert_eq!(def.name, "test_tensor");
        assert_eq!(def.dtype, TensorDataType::Float32);
        assert_eq!(def.shape.dims(), &[100]);
    }

    #[test]
    fn test_tensor_def_byte_size() {
        let def = TensorDef::new(
            "test",
            TensorDataType::Float32,
            TensorShape::batch_vector(4, 256),
        );
        // 4 * 256 elements * 4 bytes per f32 = 4096 bytes
        assert_eq!(def.byte_size(), 4096);

        let def64 = TensorDef::new(
            "test64",
            TensorDataType::Float64,
            TensorShape::vector(100),
        );
        // 100 elements * 8 bytes per f64 = 800 bytes
        assert_eq!(def64.byte_size(), 800);
    }

    // TensorData tests
    #[test]
    fn test_tensor_data_dtype() {
        let data = TensorData::Float32(vec![1.0, 2.0, 3.0]);
        assert_eq!(data.dtype(), TensorDataType::Float32);

        let data = TensorData::Int64(vec![1, 2, 3]);
        assert_eq!(data.dtype(), TensorDataType::Int64);
    }

    #[test]
    fn test_tensor_data_len() {
        let data = TensorData::Float32(vec![1.0, 2.0, 3.0]);
        assert_eq!(data.len(), 3);

        let empty_data = TensorData::Int32(vec![]);
        assert_eq!(empty_data.len(), 0);
    }

    #[test]
    fn test_tensor_data_is_empty() {
        let data = TensorData::Float32(vec![1.0]);
        assert!(!data.is_empty());

        let empty_data = TensorData::Float32(vec![]);
        assert!(empty_data.is_empty());
    }

    #[test]
    fn test_tensor_data_to_contents_float32() {
        let data = TensorData::Float32(vec![1.0, 2.0, 3.0]);
        let contents = data.to_contents();

        assert_eq!(contents.fp32_contents, vec![1.0, 2.0, 3.0]);
        assert!(contents.int_contents.is_empty());
    }

    #[test]
    fn test_tensor_data_to_contents_int32() {
        let data = TensorData::Int32(vec![10, 20, 30]);
        let contents = data.to_contents();

        assert_eq!(contents.int_contents, vec![10, 20, 30]);
        assert!(contents.fp32_contents.is_empty());
    }

    #[test]
    fn test_tensor_data_to_contents_uint8() {
        let data = TensorData::UInt8(vec![1, 2, 3]);
        let contents = data.to_contents();

        // UInt8 converted to u32
        assert_eq!(contents.uint_contents, vec![1u32, 2u32, 3u32]);
    }

    // RawTensor tests
    #[test]
    fn test_raw_tensor_as_f32_success() {
        // Create a raw tensor with 3 f32 values: 1.0, 2.0, 3.0
        let values = vec![1.0f32, 2.0f32, 3.0f32];
        let mut bytes = Vec::new();
        for val in &values {
            bytes.extend_from_slice(&val.to_le_bytes());
        }

        let def = TensorDef::new("test", TensorDataType::Float32, TensorShape::vector(3));
        let tensor = RawTensor {
            def,
            data: Bytes::from(bytes),
        };

        let result = tensor.as_f32().unwrap();
        assert_eq!(result.len(), 3);
        assert!((result[0] - 1.0).abs() < 0.001);
        assert!((result[1] - 2.0).abs() < 0.001);
        assert!((result[2] - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_raw_tensor_as_f32_wrong_type() {
        let def = TensorDef::new("test", TensorDataType::Int32, TensorShape::vector(3));
        let tensor = RawTensor {
            def,
            data: Bytes::from(vec![0u8; 12]),
        };

        let result = tensor.as_f32();
        assert!(result.is_err());
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(err_msg.contains("Cannot parse tensor"));
        assert!(err_msg.contains("as f32"));
    }

    #[test]
    fn test_raw_tensor_as_i32_success() {
        // Create a raw tensor with 3 i32 values: 10, 20, 30
        let values = vec![10i32, 20i32, 30i32];
        let mut bytes = Vec::new();
        for val in &values {
            bytes.extend_from_slice(&val.to_le_bytes());
        }

        let def = TensorDef::new("test", TensorDataType::Int32, TensorShape::vector(3));
        let tensor = RawTensor {
            def,
            data: Bytes::from(bytes),
        };

        let result = tensor.as_i32().unwrap();
        assert_eq!(result, vec![10, 20, 30]);
    }

    #[test]
    fn test_raw_tensor_as_i64_success() {
        // Create a raw tensor with 2 i64 values: 100, 200
        let values = vec![100i64, 200i64];
        let mut bytes = Vec::new();
        for val in &values {
            bytes.extend_from_slice(&val.to_le_bytes());
        }

        let def = TensorDef::new("test", TensorDataType::Int64, TensorShape::vector(2));
        let tensor = RawTensor {
            def,
            data: Bytes::from(bytes),
        };

        let result = tensor.as_i64().unwrap();
        assert_eq!(result, vec![100, 200]);
    }

    #[test]
    fn test_raw_tensor_as_scalar_f32() {
        let value = 42.5f32;
        let bytes = value.to_le_bytes();

        let def = TensorDef::new("test", TensorDataType::Float32, TensorShape::scalar());
        let tensor = RawTensor {
            def,
            data: Bytes::from(bytes.to_vec()),
        };

        let result = tensor.as_scalar_f32().unwrap();
        assert!((result - 42.5).abs() < 0.001);
    }

    #[test]
    fn test_raw_tensor_as_scalar_i64() {
        let value = 12345i64;
        let bytes = value.to_le_bytes();

        let def = TensorDef::new("test", TensorDataType::Int64, TensorShape::scalar());
        let tensor = RawTensor {
            def,
            data: Bytes::from(bytes.to_vec()),
        };

        let result = tensor.as_scalar_i64().unwrap();
        assert_eq!(result, 12345);
    }

    #[test]
    fn test_raw_tensor_as_scalar_empty() {
        let def = TensorDef::new("test", TensorDataType::Float32, TensorShape::scalar());
        let tensor = RawTensor {
            def,
            data: Bytes::from(vec![]), // Empty data
        };

        let result = tensor.as_scalar_f32();
        assert!(result.is_err());
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(err_msg.contains("is empty"));
    }

    // Test edge cases
    #[test]
    fn test_tensor_shape_zero_dim() {
        let shape = TensorShape::new(vec![0, 10]);
        // 0 * 10 = 0 elements, but we use max(1, dim) so it becomes 1 * 10 = 10
        assert_eq!(shape.num_elements(), 10);
    }

    #[test]
    fn test_tensor_def_byte_size_scalar() {
        let def = TensorDef::new("test", TensorDataType::Float32, TensorShape::scalar());
        assert_eq!(def.byte_size(), 4); // 1 element * 4 bytes
    }

    #[test]
    fn test_tensor_data_bool() {
        let data = TensorData::Bool(vec![true, false, true]);
        assert_eq!(data.dtype(), TensorDataType::Bool);
        assert_eq!(data.len(), 3);

        let contents = data.to_contents();
        assert_eq!(contents.bool_contents, vec![true, false, true]);
    }
}
