//! Performance optimization demonstration.
//!
//! This example showcases the advanced performance optimizations implemented
//! in the AMIRA Rust ASR Server, including lock-free memory pools, CPU affinity
//! management, SIMD operations, and specialized thread pools.

use amira_rust_asr_server::asr::memory::global_pools as global_lockfree_pools;
use amira_rust_asr_server::asr::simd::{
    softmax_optimized, batch_normalize_optimized, dot_product_optimized,
};
use amira_rust_asr_server::performance::{
    AffinityManager, ThreadType, SpecializedExecutor, numa_allocate_vec,
    numa_aware::global_numa_allocator,
};
use std::time::{Duration, Instant};
use tokio::time::sleep;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 AMIRA ASR Server Performance Optimization Demo");
    println!("{}", "=".repeat(50));

    // Demonstrate CPU affinity management
    demo_cpu_affinity().await?;
    
    // Demonstrate lock-free memory pools
    demo_lockfree_memory_pools().await?;
    
    // Demonstrate SIMD optimizations
    demo_simd_optimizations().await?;
    
    // Demonstrate specialized thread pools
    demo_specialized_thread_pools().await?;
    
    // Demonstrate NUMA awareness
    demo_numa_awareness().await?;
    
    println!("\n✅ Performance optimization demo completed!");
    Ok(())
}

async fn demo_cpu_affinity() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔧 CPU Affinity Management Demo");
    println!("{}", "-".repeat(30));
    
    let affinity_manager = AffinityManager::new();
    let topology = affinity_manager.topology_info();
    
    println!("CPU Topology: {}", topology);
    println!("Affinity supported: {}", affinity_manager.is_affinity_supported());
    
    // Demonstrate recommended thread counts
    for thread_type in [ThreadType::Io, ThreadType::Inference, ThreadType::Network, ThreadType::Background] {
        let count = affinity_manager.recommended_thread_count(thread_type);
        println!("{:?} threads recommended: {}", thread_type, count);
    }
    
    // Spawn a thread with specific affinity
    let handle = affinity_manager.spawn_with_affinity(
        ThreadType::Inference,
        Some("demo-inference".to_string()),
        || {
            println!("  🧵 Running inference thread with CPU affinity");
            std::thread::sleep(Duration::from_millis(100));
            "inference complete"
        },
    );
    
    let result = handle.join().unwrap();
    println!("  ✅ Thread result: {}", result);
    
    Ok(())
}

async fn demo_lockfree_memory_pools() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🏎️  Lock-Free Memory Pools Demo");
    println!("{}", "-".repeat(30));
    
    let pools = global_lockfree_pools();
    
    // Demonstrate concurrent access to memory pools
    println!("Initial pool stats:");
    println!("  {}", pools.stats());
    
    // Simulate high-concurrency workload
    let mut handles = Vec::new();
    
    for i in 0..10 {
        let handle = tokio::spawn(async move {
            for j in 0..100 {
                // Get audio buffer from lock-free pool
                let mut audio_buf = global_lockfree_pools().audio_buffers.get();
                audio_buf.push(i as f32 * 100.0 + j as f32);
                
                // Get encoder buffer
                let mut encoder_buf = global_lockfree_pools().encoder_inputs.get();
                encoder_buf.extend_from_slice(&[1.0, 2.0, 3.0, 4.0]);
                
                // Simulate some work
                tokio::task::yield_now().await;
                
                // Buffers automatically return to pool when dropped
            }
        });
        handles.push(handle);
    }
    
    // Wait for all tasks to complete
    for handle in handles {
        handle.await?;
    }
    
    println!("Pool stats after concurrent workload:");
    let final_stats = pools.stats();
    println!("  {}", final_stats);
    
    // Demonstrate hit rates
    println!("Pool efficiency:");
    println!("  Audio buffers hit rate: {:.1}%", final_stats.audio_buffers.hit_rate());
    println!("  Encoder inputs hit rate: {:.1}%", final_stats.encoder_inputs.hit_rate());
    
    Ok(())
}

async fn demo_simd_optimizations() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n⚡ SIMD Optimizations Demo");
    println!("{}", "-".repeat(30));
    
    // Demonstrate optimized softmax
    println!("Testing SIMD-optimized softmax:");
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut output = vec![0.0; input.len()];
    
    let start = Instant::now();
    softmax_optimized(&input, &mut output)?;
    let duration = start.elapsed();
    
    println!("  Input:  {:?}", input);
    println!("  Output: {:?}", output.iter().map(|x| format!("{:.4}", x)).collect::<Vec<_>>());
    println!("  Sum:    {:.6}", output.iter().sum::<f32>());
    println!("  Time:   {:?}", duration);
    
    // Demonstrate batch normalization
    println!("\nTesting SIMD-optimized batch normalization:");
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut output = vec![0.0; input.len()];
    let mean = 4.5;
    let variance = 5.25;
    let epsilon = 1e-8;
    
    let start = Instant::now();
    batch_normalize_optimized(&input, &mut output, mean, variance, epsilon)?;
    let duration = start.elapsed();
    
    println!("  Input:  {:?}", input);
    println!("  Output: {:?}", output.iter().map(|x| format!("{:.4}", x)).collect::<Vec<_>>());
    println!("  Time:   {:?}", duration);
    
    // Demonstrate dot product
    println!("\nTesting SIMD-optimized dot product:");
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b = vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
    
    let start = Instant::now();
    let result = dot_product_optimized(&a, &b)?;
    let duration = start.elapsed();
    
    println!("  Vector A: {:?}", a);
    println!("  Vector B: {:?}", b);
    println!("  Dot product: {:.2}", result);
    println!("  Time: {:?}", duration);
    
    Ok(())
}

async fn demo_specialized_thread_pools() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🎯 Specialized Thread Pools Demo");
    println!("{}", "-".repeat(30));
    
    let executor = SpecializedExecutor::new().map_err(|e| format!("Failed to create executor: {}", e))?;
    let stats = executor.stats();
    
    println!("Executor configuration:");
    println!("  {}", stats);
    
    // Demonstrate different types of tasks on specialized pools
    println!("\nSpawning tasks on specialized thread pools:");
    
    // I/O task
    let io_handle = executor.spawn_io(async {
        println!("  🔄 I/O task: Simulating file read");
        sleep(Duration::from_millis(50)).await;
        "I/O complete"
    });
    
    // Inference task
    let inference_handle = executor.spawn_inference(async {
        println!("  🧠 Inference task: Performing matrix computation");
        
        // Simulate CPU-intensive computation
        let mut matrix = vec![vec![0.0f32; 100]; 100];
        for i in 0..100 {
            for j in 0..100 {
                matrix[i][j] = (i * j) as f32;
            }
        }
        
        let sum: f32 = matrix.iter().flat_map(|row| row.iter()).sum();
        format!("Matrix sum: {:.0}", sum)
    });
    
    // Network task
    let network_handle = executor.spawn_network(async {
        println!("  🌐 Network task: Simulating gRPC call");
        sleep(Duration::from_millis(30)).await;
        "Network call complete"
    });
    
    // Wait for all tasks to complete
    let (io_result, inference_result, network_result) = 
        tokio::join!(io_handle, inference_handle, network_handle);
    
    println!("Task results:");
    println!("  {}", io_result?);
    println!("  {}", inference_result?);
    println!("  {}", network_result?);
    
    // Shutdown the executor gracefully
    executor.shutdown();
    
    Ok(())
}

async fn demo_numa_awareness() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🏗️  NUMA Awareness Demo");
    println!("{}", "-".repeat(30));
    
    let numa_allocator = global_numa_allocator();
    let topology = numa_allocator.topology();
    
    println!("NUMA configuration:");
    println!("  NUMA available: {}", topology.numa_available);
    println!("  Node count: {}", topology.node_count());
    println!("  NUMA enabled: {}", numa_allocator.is_numa_enabled());
    
    if topology.numa_available {
        for (node, cpus) in &topology.cores_per_node {
            println!("  Node {}: CPUs {:?}", node.id, cpus);
        }
    }
    
    // Demonstrate NUMA-aware allocation
    println!("\nPerforming NUMA-aware allocations:");
    
    let mut handles = Vec::new();
    for i in 0..4 {
        let handle = tokio::spawn(async move {
            // Allocate vectors with NUMA awareness
            let vec1: Vec<f32> = numa_allocate_vec(1000);
            let vec2: Vec<i32> = numa_allocate_vec(500);
            
            println!("  Task {}: Allocated vectors of {} and {} elements", i, vec1.capacity(), vec2.capacity());
            
            // Simulate some work
            sleep(Duration::from_millis(10)).await;
        });
        handles.push(handle);
    }
    
    for handle in handles {
        handle.await?;
    }
    
    // Show allocation statistics
    let stats = numa_allocator.allocation_stats();
    println!("NUMA allocation statistics:");
    for (node, count) in stats {
        println!("  Node {}: {} allocations", node.id, count);
    }
    
    Ok(())
}

// Helper function to run a benchmark
#[allow(dead_code)]
async fn benchmark_operation<F, R>(name: &str, iterations: usize, mut operation: F) -> Duration
where
    F: FnMut() -> R,
{
    println!("Benchmarking {}: {} iterations", name, iterations);
    
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = operation();
    }
    let duration = start.elapsed();
    
    let avg_duration = duration / iterations as u32;
    println!("  Total time: {:?}", duration);
    println!("  Average per operation: {:?}", avg_duration);
    
    duration
}