#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <vector>

// CPU implementation placeholder
void cpu_implementation(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// CUDA kernel placeholder
__global__ void my_kernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Utility function for CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

int main() {
    const int N = 1000000;  // Vector size
    const size_t bytes = N * sizeof(float);
    
    // Host data
    std::vector<float> h_a(N, 1.0f);
    std::vector<float> h_b(N, 2.0f);
    std::vector<float> h_c_cpu(N);
    std::vector<float> h_c_gpu(N);
    
    // Device data
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));
    
    // CPU timing
    auto start_cpu = std::chrono::high_resolution_clock::now();
    cpu_implementation(h_a.data(), h_b.data(), h_c_cpu.data(), N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration<float, std::milli>(end_cpu - start_cpu).count();
    
    // GPU timing
    cudaEvent_t start_gpu, stop_gpu;
    CUDA_CHECK(cudaEventCreate(&start_gpu));
    CUDA_CHECK(cudaEventCreate(&stop_gpu));
    
    const int block_size = 256;
    const int grid_size = (N + block_size - 1) / block_size;
    
    CUDA_CHECK(cudaEventRecord(start_gpu));
    my_kernel<<<grid_size, block_size>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaEventRecord(stop_gpu));
    CUDA_CHECK(cudaEventSynchronize(stop_gpu));
    
    float gpu_time;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time, start_gpu, stop_gpu));
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_c_gpu.data(), d_c, bytes, cudaMemcpyDeviceToHost));
    
    // Verify results
    bool correct = true;
    for (int i = 0; i < N && correct; i++) {
        if (abs(h_c_cpu[i] - h_c_gpu[i]) > 1e-5) {
            correct = false;
        }
    }
    
    // Print results
    std::cout << "Vector size: " << N << " elements" << std::endl;
    std::cout << "CPU time: " << cpu_time << " ms" << std::endl;
    std::cout << "GPU time: " << gpu_time << " ms" << std::endl;
    std::cout << "Speedup: " << cpu_time / gpu_time << "x" << std::endl;
    std::cout << "Results match: " << (correct ? "YES" : "NO") << std::endl;
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaEventDestroy(start_gpu));
    CUDA_CHECK(cudaEventDestroy(stop_gpu));
    
    return 0;
}