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
        
        // Use the triton installation from our setup script
        let install_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
        let third_party_path = std::path::Path::new(&install_dir).join("third_party");

        let triton_lib_path = third_party_path.join("tritonserver/lib");
        let triton_include_path = third_party_path.join("tritonserver/include");
        
        if !triton_lib_path.exists() || !triton_include_path.exists() {
            panic!(
                "Triton library/include path not found. Did you run 'setup_dependencies.sh'? Expected: {} and {}",
                triton_lib_path.display(),
                triton_include_path.display()
            );
        }
        
        println!("cargo:rustc-link-search=native={}", triton_lib_path.display());
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
                    &triton_include_path.to_string_lossy(),
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
