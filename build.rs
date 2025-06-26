fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Compile Triton Inference Server proto files
    tonic_build::configure()
        .build_server(false)
        .build_client(true)
        .compile(&["proto/grpc_service.proto"], &["proto"])?;
    Ok(())
}
