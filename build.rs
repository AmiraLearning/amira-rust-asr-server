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
        let project_root = std::path::Path::new(&install_dir);

        let triton_lib_path = project_root.join("lib");
        let triton_include_path = project_root.join("include");
        
        if !triton_lib_path.exists() || !triton_include_path.exists() {
            panic!(
                "Triton library/include path not found. Did you run 'setup_dependencies.sh'? Expected: {} and {}",
                triton_lib_path.display(),
                triton_include_path.display()
            );
        }
        
        println!("cargo:rustc-link-search=native={}", triton_lib_path.display());
        println!("cargo:rustc-link-search=native={}", project_root.join("local_lib").display());
        println!("cargo:rustc-link-search=native=/usr/local/lib");
        println!("cargo:rustc-link-lib=tritonserver");
        
        // Link additional Triton libraries
        println!("cargo:rustc-link-lib=tritoncommonerror");
        println!("cargo:rustc-link-lib=tritoncommonlogging");
        println!("cargo:rustc-link-lib=tritoncommonmodelconfig");
        
        // Set up runtime library path
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", triton_lib_path.display());
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", project_root.join("local_lib").display());
        
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
            
            // Add CUDA library search paths
            println!("cargo:rustc-link-search=native=/lib/x86_64-linux-gnu");
            println!("cargo:rustc-link-search=native=/usr/local/cuda/targets/x86_64-linux/lib");
            
            // Link CUDA libraries
            println!("cargo:rustc-link-lib=cudart");
            println!("cargo:rustc-link-lib=cuda");
            println!("cargo:rustc-link-lib=stdc++");
        } else {
            println!("cargo:warning=NVCC not found, CUDA features will be disabled");
        }
    }
    
    Ok(())
}
