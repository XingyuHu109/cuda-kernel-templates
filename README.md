# CUDA Kernel Templates

This repository contains minimal CUDA kernel templates to help you get started with GPU programming.

## Templates

### Vector Addition Template (`vector_add_template.cu`)

A minimal but complete CUDA template that demonstrates:
- Vector addition operation on both CPU and GPU
- Performance comparison between CPU and GPU implementations
- Proper CUDA error checking with the `CUDA_CHECK` macro
- Memory management (allocation, copying, cleanup)
- Result verification
- GPU timing using CUDA events

This template serves as a foundation for:
- Learning CUDA programming basics
- Benchmarking GPU vs CPU performance
- Starting point for more complex CUDA kernels

## Building and Running

### Prerequisites
- NVIDIA CUDA Toolkit (version 10.0 or later)
- NVIDIA GPU with compute capability 5.0 or higher
- C++11 compatible compiler

### Build Instructions

Using the provided Makefile:
```bash
# Build the template
make

# Build and run
make run

# Clean build artifacts
make clean
```

Manual compilation:
```bash
nvcc -O2 -arch=sm_50 vector_add_template.cu -o vector_add_template
./vector_add_template
```

## Template Structure

The template includes:
1. **Headers**: Standard CUDA runtime, timing, and container headers
2. **CPU Implementation**: Reference implementation for comparison
3. **CUDA Kernel**: GPU implementation of the same operation
4. **Error Checking**: Macro for robust CUDA error handling
5. **Main Function**: Complete workflow from setup to cleanup
6. **Performance Measurement**: Timing comparison between CPU and GPU
7. **Result Verification**: Ensures correctness of GPU computation

## Customizing the Template

To adapt this template for your own kernels:
1. Replace the `my_kernel` function with your custom kernel
2. Update the `cpu_implementation` with your CPU reference
3. Modify data types and sizes as needed
4. Adjust block and grid sizes for your specific use case
5. Update memory allocation and copying for your data structures

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.