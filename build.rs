fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Compile Triton Inference Server proto files
    tonic_build::configure()
        .build_server(false)
        .build_client(true)
        .compile(&["proto/grpc_service.proto"], &["proto"])?;
    
    // Compile CUDA FFI if feature is enabled
    #[cfg(feature = "cuda")]
    {
        println!("cargo:rerun-if-changed=src/cuda/cuda_helper.cu");
        println!("cargo:rerun-if-changed=build.rs");
        
        // Set up CUDA and Triton library paths
        let triton_lib_path = std::env::var("TRITON_LIB_PATH")
            .unwrap_or_else(|_| "/opt/tritonserver/lib".to_string());
        let triton_include_path = std::env::var("TRITON_INCLUDE_PATH")
            .unwrap_or_else(|_| "/opt/tritonserver/include".to_string());
        
        println!("cargo:rustc-link-search=native={}", triton_lib_path);
        println!("cargo:rustc-link-lib=tritonserver");
        
        // Use nvcc to compile CUDA code
        let nvcc_output = std::process::Command::new("nvcc")
            .arg("--version")
            .output();
        
        if nvcc_output.is_ok() {
            println!("cargo:rustc-env=CUDA_AVAILABLE=1");
            
            // Compile CUDA helper
            let cuda_obj = std::process::Command::new("nvcc")
                .args([
                    "-c",
                    "src/cuda/cuda_helper.cu",
                    "-o",
                    "cuda_helper.o",
                    "--compiler-options",
                    "-fPIC",
                    "-I",
                    &triton_include_path,
                    "-std=c++17",
                ])
                .output()
                .expect("Failed to compile CUDA helper");
            
            if !cuda_obj.status.success() {
                panic!("CUDA compilation failed: {}", String::from_utf8_lossy(&cuda_obj.stderr));
            }
            
            // Link the compiled CUDA object file
            println!("cargo:rustc-link-arg=cuda_helper.o");
            println!("cargo:rustc-link-lib=cudart");
            println!("cargo:rustc-link-lib=cuda");
        } else {
            println!("cargo:warning=NVCC not found, CUDA features will be disabled");
        }
    }
    
    Ok(())
}
